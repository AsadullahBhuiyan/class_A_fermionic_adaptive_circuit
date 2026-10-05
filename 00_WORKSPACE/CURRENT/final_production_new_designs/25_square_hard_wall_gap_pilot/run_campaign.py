"""Seven independent square-system, ten-cycle covariance tasks. No legacy imports."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'src'))
from classA_U1FGTN_gpu import classA_U1FGTN_gpu
from endpoint_spectrum import extract_endpoint

SOURCE_FILES = ('run_campaign.py', 'endpoint_spectrum.py', 'src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py')


def default_config():
    return dict(sampling_revision='square_hard_wall_gap_l20-44_s10_t10_v1', root_seed=2026092725,
                sizes=[20,24,28,32,36,40,44], samples_per_case=10, cycles=10,
                execution_batch_size=10, result_shard_size=5, segment_cycles=5,
                DW=True, dw_truncation=True, meas_slab_only=True, triv_region_local_mode=False,
                alpha_1=1., alpha_2=30., nshell=1, trial_orbitals='X', filling_fraction=.5,
                init_mode='maxmix', n_a=.5, sequence='raster_y', perfect_correction=True,
                postselect=False, state_representation='covariance', dtype='complex128',
                device='cuda:0', backend='local', cap_tolerance=1e-9,
                initial_purity_tolerance=.50000001, gpu_allocator_limit_bytes=35_000_000_000,
                spectrum_cycles=[10], observer_sample_chunk=1, covariance_spectral_clip=False)


@dataclass(frozen=True)
class Task:
    L: int
    samples: int = 10
    cycles: int = 10
    root_seed: int = 2026092725

    @property
    def name(self):
        return f'L{self.L:03d}_samples000-{self.samples-1:03d}_T{self.cycles:03d}'

    @property
    def seed(self):
        return int.from_bytes(hashlib.sha256(f'{self.root_seed}|{self.name}'.encode()).digest()[:4], 'little')

    @property
    def active_modes(self):
        return self.L*(self.L+2)


def tasks(config):
    return [Task(l, config['samples_per_case'], config['cycles'], config['root_seed'])
            for l in sorted(config['sizes'], reverse=True)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(task, config):
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return dict(schema='square_gap_v1', task=task.name, seed=task.seed,
                config_sha256=hashlib.sha256(canonical.encode()).hexdigest(), configuration=config,
                source_hashes={name: sha(HERE/name) for name in SOURCE_FILES},
                canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit')


def result_paths(output, task, start):
    path = Path(output)/'results'/task.name/f'samples_{start:03d}-{min(start+5,task.samples)-1:03d}.npz'
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


def result_verified(output, task, start, ident):
    path, receipt = result_paths(output, task, start)
    ids = list(range(start, min(start+5,task.samples)))
    if not pair_verified(path, receipt, dict(ident, kind='result', sample_indices=ids)):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return (np.array_equal(z['sample_indices'], ids) and int(z['T']) == task.cycles and
                    int(z['Nx']) == task.L and int(z['Ny']) == task.L and
                    z['occupation_spectrum_raw'].shape == (len(ids),task.active_modes) and
                    np.isfinite(z['occupation_spectrum_raw']).all() and
                    np.array_equal(z['lyapunov_gap'], z['modular_gap']/(2*task.cycles)))
    except (OSError, ValueError, KeyError):
        return False


def publish_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name+'.tmp')
    shutil.copyfile(source, temp)
    if temp.stat().st_size != source.stat().st_size or sha(temp) != sha(source):
        raise OSError(f'DriveFS temporary readback failed: {temp}')
    os.replace(temp, destination)
    if destination.stat().st_size != source.stat().st_size or sha(destination) != sha(source):
        raise OSError(f'DriveFS final readback failed: {destination}')


def publish_pair(payload, path, receipt, expected, scratch, compressed):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch/path.name
    saver = np.savez_compressed if compressed else np.savez
    saver(local, **payload)
    publish_file(local, path)
    record = dict(expected, filename=path.name, bytes=path.stat().st_size, sha256=sha(path))
    local_json = scratch/receipt.name
    local_json.write_text(json.dumps(record, indent=2)+'\n')
    publish_file(local_json, receipt)
    if not pair_verified(path, receipt, expected):
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
                   sample_indices=np.arange(task.samples), **rng)
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
        if (cycle not in range(segment, task.cycles+1, segment) or
            record['completed_cycle'] != cycle or
            payload['G'].shape != (task.samples, 2*task.L**2, 2*task.L**2) or
            payload['G'].dtype != np.complex128 or not np.isfinite(payload['G']).all() or
            not np.array_equal(payload['sample_indices'], np.arange(task.samples))):
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
    model = classA_U1FGTN_gpu(Nx=task.L, Ny=task.L, DW=True, nshell=1,
                             filling_frac=.5, alpha_1=1., alpha_2=30., trial_orbitals='X',
                             dw_truncation=True, triv_region_local_mode=False,
                             device=device, dtype='complex128', backend='local')
    if tuple(model.DW_loc) != (task.L//4, 3*task.L//4):
        raise ValueError('Unexpected wall locations')
    active = model.active_top_layer_indices(meas_slab_only=True)
    if active.numel() != task.active_modes or model.dtype != torch.complex128:
        raise ValueError('Active-slab/dtype mismatch')
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
        n_a=.5, sequence='raster_y', meas_slab_only=True, batch_size=task.samples,
        return_data=True, state_representation='covariance',
        initial_purity_tolerance=.50000001, cycle_observer=observe)
    if result.get('state_representation_resolved') != 'covariance':
        raise RuntimeError('Canonical engine did not use covariance representation')
    if bool(result.get('G_init_prepared')) != bool(completed):
        raise RuntimeError('Prepared-state continuation flag not honored')
    if bool(result.get('exterior_preparation_performed')) != (completed == 0):
        raise RuntimeError('Exterior preparation must occur exactly once')
    return np.asarray(result['G_final']), capture_rng()


def cleanup_checkpoint(output, task, ident):
    if not all(result_verified(output, task, start, ident) for start in range(0,task.samples,5)):
        raise RuntimeError('Cannot remove checkpoint before every result verifies')
    for path in checkpoint_paths(output, task):
        path.unlink(missing_ok=True)
    print(f'[cleanup] removed rolling checkpoint for {task.name}; verified endpoint spectra remain', flush=True)


def run_task(task, config, output, scratch, outer=None, device=None):
    ident = identity(task, config)
    starts = list(range(0,task.samples,5))
    if all(result_verified(output, task, start, ident) for start in starts):
        cleanup_checkpoint(output, task, ident)
        return
    checkpoint = load_checkpoint(output, task, ident, config['segment_cycles'])
    completed = 0 if checkpoint is None else int(checkpoint['completed_cycle'])
    G = None if checkpoint is None else checkpoint['G']
    rng = None if checkpoint is None else {k:v for k,v in checkpoint.items() if k.startswith('rng_')}
    elapsed = 0. if checkpoint is None else float(checkpoint['elapsed_seconds'])
    device = device or config['device']
    print(f'[task] {task.name}, seed={task.seed}, resume cycle={completed}; walls={task.L//4,3*task.L//4}', flush=True)
    if checkpoint is None and any(p.exists() for p in checkpoint_paths(output, task)):
        print('[warning] Invalid/partial checkpoint rejected; deterministic restart from cycle zero.', flush=True)
    model = None
    if completed < task.cycles:
        model = build_model(task, device)
        if completed == 0:
            np.random.seed(task.seed); torch.manual_seed(task.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task.seed)
        with tqdm(total=task.cycles, initial=completed, desc=f'L={task.L} physical cycles', unit='cycle') as bar:
            while completed < task.cycles:
                count = min(config['segment_cycles'], task.cycles-completed)
                begin = time.perf_counter()
                G, rng = run_segment(model, task, G, completed, count, rng, bar)
                duration = time.perf_counter()-begin
                elapsed += duration
                completed += count
                checkpoint_start = time.perf_counter()
                save_checkpoint(output, scratch, task, ident, G, completed, elapsed, rng)
                print(f'[timing] L={task.L}: {duration/count:.2f}s/cycle; checkpoint {time.perf_counter()-checkpoint_start:.2f}s', flush=True)
    # Read active indices from the engine even when resuming an endpoint-only task.
    if model is None:
        model = build_model(task, device)
    active = model.active_top_layer_indices(meas_slab_only=True).detach().cpu().numpy()
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    print(f'[endpoint] diagonalizing {task.samples} active spectra, {task.active_modes} modes each', flush=True)
    begin = time.perf_counter()
    products = extract_endpoint(G, active, task.cycles, device=device)
    endpoint_seconds = time.perf_counter()-begin
    for start in starts:
        if result_verified(output, task, start, ident):
            continue
        stop = min(start+5,task.samples)
        path, receipt = result_paths(output, task, start)
        payload = {key:values[start:stop] for key,values in products.items()}
        payload.update(sample_indices=np.arange(start,stop), active_indices=active,
                       Nx=np.array(task.L), Ny=np.array(task.L), T=np.array(task.cycles),
                       walls=np.array([task.L//4,3*task.L//4]), seed=np.array(task.seed),
                       dynamics_seconds=np.array(elapsed), endpoint_seconds=np.array(endpoint_seconds),
                       configuration_json=np.array(json.dumps(config,sort_keys=True)),
                       source_hashes_json=np.array(json.dumps(ident['source_hashes'],sort_keys=True)))
        publish_pair(payload, path, receipt, dict(ident,kind='result',sample_indices=list(range(start,stop))),
                     Path(scratch)/task.name, compressed=True)
        if not result_verified(output, task, start, ident):
            raise OSError('Endpoint result did not verify')
        if outer is not None:
            outer.update(1)
            outer.set_postfix(completed=outer.n, pending=14-outer.n, failed=0)
    cleanup_checkpoint(output, task, ident)
    if str(device).startswith('cuda'):
        print(f'[memory] peak reserved {torch.cuda.max_memory_reserved()/1e9:.3f} GB', flush=True)
    print(f'[complete] {task.name}: dynamics {elapsed:.1f}s; endpoint {endpoint_seconds:.1f}s', flush=True)


def inventory(config, output):
    rows = []
    for task in tasks(config):
        ident = identity(task,config)
        done = sum(result_verified(output,task,s,ident) for s in range(0,task.samples,5))
        checkpoint = load_checkpoint(output,task,ident,config['segment_cycles']) if done < 2 else None
        rows.append(dict(task=task.name,L=task.L,completed_shards=done,
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
        parser.error('Configuration differs from the locked square-system pilot contract')
    if args.max_new_execution_batches is not None and args.max_new_execution_batches < 0:
        parser.error('Maximum new batches must be nonnegative')
    rows = inventory(config,args.output_root)
    done = sum(row['completed_shards'] for row in rows)
    print(json.dumps(dict(config=config,source_root=str(HERE),source_hashes=identity(tasks(config)[0],config)['source_hashes'],
                          output_root=str(args.output_root),scratch_root=str(args.scratch_root),
                          canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit',
                          trajectories=70,execution_batches=7,total_shards=14,completed=done,pending=14-done,
                          inventory=rows),indent=2),flush=True)
    if args.report_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Production requires an A100 40-GB-class CUDA runtime')
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
    with tqdm(total=14,initial=done,desc='durable endpoint shards',unit='shard') as bar:
        bar.set_postfix(completed=done, skipped=done, pending=14-done, failed=0)
        for task,row in zip(tasks(config),rows):
            if row['completed_shards'] == 2:
                if any(p.exists() for p in checkpoint_paths(args.output_root, task)):
                    cleanup_checkpoint(args.output_root, task, identity(task, config))
                continue
            if args.max_new_execution_batches is not None and launched >= args.max_new_execution_batches:
                break
            torch.cuda.reset_peak_memory_stats()
            try:
                run_task(task,config,args.output_root,args.scratch_root,bar)
            except Exception:
                bar.set_postfix(completed=bar.n, skipped=done, pending=14-bar.n, failed=1)
                raise
            bar.set_postfix(completed=bar.n, skipped=done, pending=14-bar.n, failed=0)
            launched += 1
            torch.cuda.empty_cache()
    print('[summary] '+json.dumps(inventory(config,args.output_root)),flush=True)


if __name__ == '__main__':
    main()
