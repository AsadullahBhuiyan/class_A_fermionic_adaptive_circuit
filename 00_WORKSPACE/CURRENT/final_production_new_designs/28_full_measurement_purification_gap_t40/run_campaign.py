"""Full-measurement purification: cycle-resolved finite modes, no legacy imports."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'src'))
from classA_U1FGTN_gpu import classA_U1FGTN_gpu
from endpoint_spectrum import extract_endpoint, validate_finite_modes, SpectrumBatcher

SPECTRUM_BATCHER = SpectrumBatcher()

SOURCE_FILES = ('run_campaign.py', 'endpoint_spectrum.py', 'src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py')


def default_config():
    return dict(sampling_revision='full_measurement_gap_nx20_ny30-60_s100_t40_cycles6-40_v2_batched', root_seed=2026092828,
                Nx=20, Ny_values=[30,36,42,48,54,60], samples_per_case=100, cycles=40,
                execution_batch_size_by_Ny={'30':100,'36':50,'42':50,'48':25,'54':25,'60':20},
                result_shard_size=5, segment_cycles=5,
                DW=True, dw_truncation=True, meas_slab_only=False, triv_region_local_mode=False,
                alpha_1=1., alpha_2=30., nshell=1, trial_orbitals='X', filling_fraction=.5,
                init_mode='maxmix', n_a=.5, sequence='raster_y', perfect_correction=True,
                postselect=False, state_representation='covariance', dtype='complex128',
                device='cuda:0', backend='local', cap_tolerance=1e-9,
                initial_purity_tolerance=.50000001, gpu_allocator_limit_bytes=35_000_000_000,
                spectrum_cycles=list(range(6,41)), observer_matrix_batches=[1,2,5], observer_headroom_gib=8, covariance_spectral_clip=False,
                cycle_finite_mode_vectors=True, finite_mode_basis='full_centered_covariance_eigenvectors')


@dataclass(frozen=True)
class Task:
    Ny: int
    Nx: int = 20
    samples: int = 10
    sample_start: int = 0
    cycles: int = 40
    root_seed: int = 2026092828

    @property
    def name(self):
        return f'Nx{self.Nx:03d}_Ny{self.Ny:03d}_samples{self.sample_start:03d}-{self.sample_start+self.samples-1:03d}_T{self.cycles:03d}'

    @property
    def seed(self):
        return int.from_bytes(hashlib.sha256(f'{self.root_seed}|{self.name}'.encode()).digest()[:4], 'little')

    @property
    def active_modes(self):
        return 2*self.Nx*self.Ny


def tasks(config):
    result = []
    for ny in sorted(config['Ny_values'], reverse=True):
        batch = config['execution_batch_size_by_Ny'][str(ny)]
        for start in range(0,config['samples_per_case'],batch):
            result.append(Task(Ny=ny, Nx=config['Nx'], sample_start=start,
                               samples=min(batch,config['samples_per_case']-start),
                               cycles=config['cycles'], root_seed=config['root_seed']))
    return result


def slots(task, config):
    return [(cycle,start) for cycle in config['spectrum_cycles'] if cycle <= task.cycles
            for start in range(0,task.samples,5)]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(task, config):
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return dict(schema='full_measurement_cycle_modes_v2', task=task.name, seed=task.seed,
                config_sha256=hashlib.sha256(canonical.encode()).hexdigest(), configuration=config,
                source_hashes={name: sha(HERE/name) for name in SOURCE_FILES},
                canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit')


def result_paths(output, task, start, cycle):
    first = task.sample_start+start
    last = task.sample_start+min(start+5,task.samples)-1
    path = Path(output)/'results'/task.name/f'cycle_{cycle:03d}'/f'samples_{first:03d}-{last:03d}.npz'
    return path, path.with_suffix('.complete.json')

def checkpoint_paths(output, task):
    path = Path(output)/'checkpoints'/task.name/'checkpoint.npz'
    return path, path.with_suffix('.json')


def pair_verified(path, receipt, expected):
    try:
        row = json.loads(receipt.read_text())
        return (all(row.get(key) == value for key,value in expected.items()) and
                row['filename'] == path.name and row['bytes'] == path.stat().st_size and
                row['sha256'] == sha(path))
    except (OSError, ValueError, KeyError):
        return False


def result_verified(output, task, start, ident, cycle):
    path, receipt = result_paths(output, task, start, cycle)
    ids = list(range(task.sample_start+start, task.sample_start+min(start+5,task.samples)))
    if not pair_verified(path, receipt, dict(ident, kind='result', cycle=cycle, sample_indices=ids)):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            validate_finite_modes(z)
            active = z['active_indices']
            expected = np.arange(task.active_modes)
            if not np.array_equal(active, expected):
                return False
            coordinates = np.column_stack(((active//2)%task.Nx, active//(2*task.Nx), active%2))
            if not np.array_equal(z['active_coordinates_x_y_orbital'], coordinates):
                return False
            return (np.array_equal(z['sample_indices'], ids) and int(z['T']) == cycle and
                    int(z['Nx']) == task.Nx and int(z['Ny']) == task.Ny and
                    z['occupation_spectrum_raw'].shape == (len(ids),task.active_modes) and
                    np.isfinite(z['occupation_spectrum_raw']).all() and
                    np.array_equal(z['lyapunov_gap'], z['modular_gap']/(2*cycle)))
    except (OSError, ValueError, KeyError, AssertionError):
        return False


def publish_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name+'.tmp')
    expected = dict(bytes=source.stat().st_size,sha256=sha(source))
    shutil.copyfile(source,temp)
    if temp.stat().st_size != expected['bytes'] or sha(temp) != expected['sha256']:
        raise OSError(f'DriveFS temporary readback failed: {temp}')
    os.replace(temp,destination)
    if destination.stat().st_size != expected['bytes']:
        raise OSError(f'DriveFS final size mismatch: {destination}')
    return expected


def save_fast_npz(path, payload):
    # Level one compresses structural zero padding cheaply; array bytes are lossless.
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for name,value in payload.items():
            with archive.open(name+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,np.asanyarray(value),allow_pickle=False)


def publish_pair(payload, path, receipt, expected, scratch, compressed):
    scratch = Path(scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    local = scratch/path.name
    if compressed:
        save_fast_npz(local,payload)
    else:
        np.savez(local,**payload)
    verified = publish_file(local,path)
    record = dict(expected,filename=path.name,**verified)
    local_json = scratch/receipt.name
    local_json.write_text(json.dumps(record,indent=2)+'\n')
    publish_file(local_json,receipt)
    if not pair_verified(path,receipt,expected):
        raise OSError('Published pair failed final verification')


def capture_rng():
    state = np.random.get_state()
    payload = dict(rng_np_algorithm=np.array(state[0]), rng_np_state=state[1],
                   rng_np_position=np.array(state[2]), rng_np_has_gauss=np.array(state[3]),
                   rng_np_gauss=np.array(state[4]), rng_torch=torch.get_rng_state().numpy(),
                   rng_cuda_count=np.array(torch.cuda.device_count() if torch.cuda.is_available() else 0))
    if torch.cuda.is_available():
        for i,state in enumerate(torch.cuda.get_rng_state_all()):
            payload[f'rng_cuda_{i}'] = state.cpu().numpy()
    return payload


def restore_rng(payload):
    np.random.set_state((str(payload['rng_np_algorithm']), payload['rng_np_state'],
                        int(payload['rng_np_position']), int(payload['rng_np_has_gauss']),
                        float(payload['rng_np_gauss'])))
    torch.set_rng_state(torch.as_tensor(payload['rng_torch'], dtype=torch.uint8).cpu())
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count != int(payload['rng_cuda_count']):
        raise ValueError('CUDA RNG device count differs from checkpoint')
    if count:
        torch.cuda.set_rng_state_all([torch.as_tensor(payload[f'rng_cuda_{i}'],dtype=torch.uint8).cpu()
                                     for i in range(count)])


def save_checkpoint(output, scratch, task, ident, G, cycle, elapsed, rng):
    path, receipt = checkpoint_paths(output, task)
    payload = dict(G=G, completed_cycle=np.array(cycle), elapsed_seconds=np.array(elapsed),
                   sample_indices=np.arange(task.sample_start,task.sample_start+task.samples), **rng)
    publish_pair(payload, path, receipt, dict(ident, kind='checkpoint', completed_cycle=cycle),
                 Path(scratch)/task.name, compressed=False)
    print(f'[checkpoint] {task.name}: verified cycle {cycle}/{task.cycles}', flush=True)


def load_checkpoint(output, task, ident, segment=5):
    path, receipt = checkpoint_paths(output, task)
    if not pair_verified(path, receipt, dict(ident, kind='checkpoint')):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            payload = {key:z[key] for key in z.files}
        cycle = int(payload['completed_cycle'])
        record = json.loads(receipt.read_text())
        if ((cycle not in range(segment, task.cycles+1, segment) and cycle != task.cycles) or
            record['completed_cycle'] != cycle or
            payload['G'].shape != (task.samples, 2*task.Nx*task.Ny, 2*task.Nx*task.Ny) or
            payload['G'].dtype != np.complex128 or not np.isfinite(payload['G']).all() or
            not np.array_equal(payload['sample_indices'], np.arange(task.sample_start,task.sample_start+task.samples))):
            return None
        required = ('rng_np_algorithm','rng_np_state','rng_np_position','rng_np_has_gauss',
                    'rng_np_gauss','rng_torch','rng_cuda_count')
        if not all(key in payload for key in required):
            return None
        if not all(f'rng_cuda_{i}' in payload for i in range(int(payload['rng_cuda_count']))):
            return None
        return payload
    except (ValueError, KeyError, OSError):
        return None


def build_model(task, device):
    model = classA_U1FGTN_gpu(Nx=task.Nx, Ny=task.Ny, DW=True, nshell=1,
                             filling_frac=.5, alpha_1=1., alpha_2=30., trial_orbitals='X',
                             dw_truncation=True, triv_region_local_mode=False,
                             device=device, dtype='complex128', backend='local')
    if tuple(model.DW_loc) != (task.Nx//4, 3*task.Nx//4):
        raise ValueError('Unexpected wall locations')
    active = model.active_top_layer_indices(meas_slab_only=False)
    if active.numel() != task.active_modes or model.dtype != torch.complex128:
        raise ValueError('Full-system/dtype mismatch')
    return model


def run_segment(model, task, G, completed, count, rng, progress_bar=None):
    def observe(cycle, **unused):
        if cycle > 0 and progress_bar is not None:
            progress_bar.update(1)
    if rng is not None:
        restore_rng(rng)  # last operation before the canonical dynamics call
    result = model.run_markov_circuit(
        G_history=False, progress=False, cycles=count, samples=task.samples,
        init_mode='maxmix', G_init=G, G_init_prepared=bool(completed),
        perfect_correction=True, postselect=False, save=False, save_init=False,
        n_a=.5, sequence='raster_y', meas_slab_only=False, batch_size=task.samples,
        return_data=True, state_representation='covariance',
        initial_purity_tolerance=.50000001, cycle_observer=observe)
    if result.get('state_representation_resolved') != 'covariance':
        raise RuntimeError('Canonical engine did not use covariance representation')
    if bool(result.get('G_init_prepared')) != bool(completed):
        raise RuntimeError('Prepared-state continuation flag not honored')
    if bool(result.get('exterior_preparation_performed')):
        raise RuntimeError('Full measurement must never perform slab exterior preparation')
    return np.asarray(result['G_final']), capture_rng()


def cleanup_checkpoint(output, task, ident):
    config = ident['configuration']
    if not all(result_verified(output,task,start,ident,cycle) for cycle,start in slots(task,config)):
        raise RuntimeError('Cannot remove checkpoint before every cycle result verifies')
    for path in checkpoint_paths(output, task):
        path.unlink(missing_ok=True)
    print(f'[cleanup] {task.name}: removed rolling state; all cycle spectra remain', flush=True)


def publish_cycle(G, cycle, task, config, ident, output, scratch, valid, outer, device, elapsed):
    started = time.perf_counter()
    active = np.arange(task.active_modes)
    coordinates = np.column_stack(((active//2)%task.Nx,active//(2*task.Nx),active%2))
    print(f'[spectra] {task.name}: cycle {cycle}, full dimension {task.active_modes}',flush=True)
    for start in range(0,task.samples,5):
        if (cycle,start) in valid:
            continue
        stop = min(start+5,task.samples)
        begin = time.perf_counter()
        if str(device).startswith('cuda'):
            products,batching = SPECTRUM_BATCHER.extract(G[start:stop],active,cycle,device)
        else:
            products = extract_endpoint(G[start:stop],active,cycle,device=device)
            batching = dict(selected=1,device='CPU test')
        products['spectral_batching_json'] = np.array(json.dumps(batching,sort_keys=True))
        ids = np.arange(task.sample_start+start,task.sample_start+stop)
        products.update(sample_indices=ids,active_indices=active,
                        active_coordinates_x_y_orbital=coordinates,
                        Nx=np.array(task.Nx),Ny=np.array(task.Ny),T=np.array(cycle),
                        walls=np.array([task.Nx//4,3*task.Nx//4]),seed=np.array(task.seed),
                        dynamics_seconds=np.array(elapsed),
                        spectral_seconds=np.array(time.perf_counter()-begin),
                        configuration_json=np.array(json.dumps(config,sort_keys=True)),
                        source_hashes_json=np.array(json.dumps(ident['source_hashes'],sort_keys=True)))
        path,receipt = result_paths(output,task,start,cycle)
        publish_pair(products,path,receipt,dict(ident,kind='result',cycle=cycle,sample_indices=ids.tolist()),
                     Path(scratch)/task.name,compressed=True)
        if not result_verified(output,task,start,ident,cycle):
            raise OSError('Cycle result failed readback')
        valid.add((cycle,start))
        # Published data are durable; do not accumulate local mode histories.
        for local in (Path(scratch)/task.name/path.name,Path(scratch)/task.name/receipt.name):
            local.unlink(missing_ok=True)
        if outer is not None:
            outer.update(1)
            outer.set_postfix(completed=outer.n,pending=outer.total-outer.n,failed=0)
        print(f'[saved] cycle {cycle}, samples {ids[0]}–{ids[-1]}, {path.stat().st_size/1e6:.1f} MB; '
              f'spectra+publication {time.perf_counter()-begin:.1f}s',flush=True)

    return time.perf_counter()-started


def run_task(task, config, output, scratch, outer=None, device=None):
    ident = identity(task,config)
    valid = {(cycle,start) for cycle,start in slots(task,config)
             if result_verified(output,task,start,ident,cycle)}
    if len(valid) == len(slots(task,config)):
        cleanup_checkpoint(output,task,ident)
        return
    checkpoint = load_checkpoint(output,task,ident,config['segment_cycles'])
    if checkpoint is not None:
        cycle = int(checkpoint['completed_cycle'])
        if any((t,s) not in valid for t,s in slots(task,config) if t < cycle):
            print('[warning] Earlier cycle result missing/corrupt: deterministic restart; valid files retained.',flush=True)
            checkpoint = None
    completed = 0 if checkpoint is None else int(checkpoint['completed_cycle'])
    G = None if checkpoint is None else checkpoint['G']
    rng = None if checkpoint is None else {k:v for k,v in checkpoint.items() if k.startswith('rng_')}
    elapsed = 0. if checkpoint is None else float(checkpoint['elapsed_seconds'])
    device = device or config['device']
    print(f'[task] {task.name}; seed={task.seed}; resume={completed}; full measurement; no exterior preparation',flush=True)
    # Construct before restoring RNG. Observer work must not affect trajectory RNG.
    model = build_model(task,device) if completed < task.cycles else None
    if completed == 0:
        np.random.seed(task.seed); torch.manual_seed(task.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(task.seed)
    checkpoint_seconds = 0.
    with tqdm(total=task.cycles,initial=completed,desc=f'Ny={task.Ny} physical cycles',unit='cycle') as bar:
        while True:
            # Spectra are durable every cycle; state checkpoints every five cycles bound replay.
            if completed in config['spectrum_cycles']:
                spectral_seconds = publish_cycle(G,completed,task,config,ident,output,scratch,valid,outer,device,elapsed)
                if spectral_seconds > .1 and completed:
                    remaining = (task.cycles-completed)*(elapsed/completed + spectral_seconds +
                                                          checkpoint_seconds/config['segment_cycles'])
                    print(f'[ETA] current batch ~{remaining/3600:.2f} h remaining '
                          '(measured compute + latest spectral/Drive stage)',flush=True)
            if completed == task.cycles:
                break
            begin = time.perf_counter()
            G,rng = run_segment(model,task,G,completed,1,rng,bar)
            elapsed += time.perf_counter()-begin
            completed += 1
            if completed % config['segment_cycles'] == 0 or completed == task.cycles:
                checkpoint_start = time.perf_counter()
                save_checkpoint(output,scratch,task,ident,G,completed,elapsed,rng)
                checkpoint_seconds = time.perf_counter()-checkpoint_start
            print(f'[timing] cycle {completed}: dynamics + state publication {time.perf_counter()-begin:.1f}s',flush=True)
    cleanup_checkpoint(output,task,ident)
    for local in (Path(scratch)/task.name/'checkpoint.npz',Path(scratch)/task.name/'checkpoint.json'):
        local.unlink(missing_ok=True)
    print(f'[complete] {task.name}: {len(valid)} verified cycle shards, dynamics {elapsed:.1f}s',flush=True)

def inventory(config, output):
    rows = []
    for task in tasks(config):
        ident = identity(task,config)
        done = sum(result_verified(output,task,s,ident,t) for t,s in slots(task,config))
        checkpoint = load_checkpoint(output,task,ident) if done < len(slots(task,config)) else None
        rows.append(dict(task=task.name,Nx=task.Nx,Ny=task.Ny,completed_shards=done,
                         total_shards=len(slots(task,config)),
                         recoverable_cycle=None if checkpoint is None else int(checkpoint['completed_cycle'])))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=HERE/'campaign_config.json')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--scratch-root',type=Path,required=True)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--max-new-execution-batches',type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config != default_config():
        parser.error('Configuration differs from the locked full-measurement cycle-spectrum contract')
    if args.max_new_execution_batches is not None and args.max_new_execution_batches < 0:
        parser.error('Maximum new batches must be nonnegative')
    rows = inventory(config,args.output_root)
    done = sum(row['completed_shards'] for row in rows)
    total = sum(row['total_shards'] for row in rows)
    print(json.dumps(dict(config=config,source_root=str(HERE),source_hashes=identity(tasks(config)[0],config)['source_hashes'],
                          output_root=str(args.output_root),scratch_root=str(args.scratch_root),
                          canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit',
                          trajectories=sum(t.samples for t in tasks(config)),execution_batches=len(tasks(config)),total_shards=total,completed=done,pending=total-done,
                          inventory=rows),indent=2),flush=True)
    if args.report_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. In Colab select Runtime > Change runtime type > A100 GPU, reconnect, and rerun from the top. If already selected, check torch.cuda.is_available() and nvidia-smi.')
    props = torch.cuda.get_device_properties(torch.device(config['device']))
    if 'A100' not in props.name or props.total_memory < 35*1024**3:
        raise RuntimeError(f'Expected A100 40-GB-class GPU; found {props.name}')
    torch.cuda.set_per_process_memory_fraction(config['gpu_allocator_limit_bytes']/props.total_memory)
    print(f'[device] {props.name}; total {props.total_memory/1e9:.2f} GB; allocator limit 35 GB',flush=True)
    for path in (args.output_root,args.scratch_root):
        path.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(path).free < 8*1024**3:
            raise OSError(f'Need 8 GiB free for temporary and rolling checkpoint copies: {path}')
    launched = 0
    with tqdm(total=total,initial=done,desc='durable cycle shards',unit='shard') as bar:
        bar.set_postfix(completed=done, skipped=done, pending=total-done, failed=0)
        for task,row in zip(tasks(config),rows):
            if row['completed_shards'] == row['total_shards']:
                if any(p.exists() for p in checkpoint_paths(args.output_root, task)):
                    cleanup_checkpoint(args.output_root, task, identity(task, config))
                continue
            if args.max_new_execution_batches is not None and launched >= args.max_new_execution_batches:
                break
            torch.cuda.reset_peak_memory_stats()
            try:
                run_task(task,config,args.output_root,args.scratch_root,bar)
            except Exception:
                bar.set_postfix(completed=bar.n, skipped=done, pending=total-bar.n, failed=1)
                raise
            bar.set_postfix(completed=bar.n, skipped=done, pending=total-bar.n, failed=0)
            launched += 1
            torch.cuda.empty_cache()
    print('[summary] '+json.dumps(inventory(config,args.output_root)),flush=True)


if __name__ == '__main__':
    main()
