"""Three disjoint alpha lanes; canonical batched covariance dynamics and exact resume."""
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
from endpoint_spectrum import extract_endpoint, validate_products

SOURCE_FILES = ('run_campaign.py', 'endpoint_spectrum.py', 'src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py')

# Exact deployment from the reported October 1 endpoint-ordering failure.
# Eigenpair sorting changes no dynamics, seeds, caps, or already sorted results.
# Accept this one audited source tuple, not arbitrary historical source changes.
PRE_SORT_SOURCE_HASHES = {
    'run_campaign.py': '0e80487e6328723f0ff90d22680e27f653b21e5750f764f028b9a06b4044fc8c',
    'endpoint_spectrum.py': 'bd90152002ff8f39fef2d07adcb2bea192db0cbd37a8cc410cb6438c9ab3709a',
    'src/classA_U1FGTN_gpu.py': '86ad0d5a40aab9cfcc63d9688b024948fed7a891309cd4376745206e2883687c',
    'src/occupied_frame_gpu.py': 'bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0',
}
S10_SOURCE_HASHES = dict(PRE_SORT_SOURCE_HASHES, **{
    'run_campaign.py': 'abe5c3a87a01537d39e64177c435e98b8afaf3dbf031696a556643da5826e08a',
    'endpoint_spectrum.py': '30f5cff64732e3fed49e81c74df6b6ea3aa596ec3833900fb5c021e0893a0b99',
})


def compatible_sources(saved, expected):
    if saved == expected:
        return True
    return (saved in (PRE_SORT_SOURCE_HASHES, S10_SOURCE_HASHES) and isinstance(expected, dict) and
            all(expected.get(k) == v for k,v in PRE_SORT_SOURCE_HASHES.items() if k.startswith('src/')))


ALPHAS = [1,1.2,1.4,1.6,1.7,1.8,1.85,1.9,1.95,1.975,2,2.025,2.05,2.1,2.15,2.2,2.3,2.4,2.6,2.8,3]

def default_config():
    """Immutable original S10 contract, retained for historical analysis."""
    return dict(sampling_revision='hard_wall_alpha21_nx20_ny30_s10_t60_endpoint_modes_v1', root_seed=2026100131,
                Nx=20, Ny=30, alpha_values=ALPHAS.copy(), samples_per_case=10, cycles=60,
                execution_batch_size=10, result_shard_size=5, segment_cycles=10,
                DW=True, dw_truncation=True, meas_slab_only=True, triv_region_local_mode=False,
                alpha_2=30., nshell=1, trial_orbitals='X', filling_fraction=.5,
                init_mode='maxmix', n_a=.5, sequence='raster_y', perfect_correction=True,
                postselect=False, state_representation='covariance', dtype='complex128',
                device='cuda:0', backend='local', cap_tolerance=1e-9,
                initial_purity_tolerance=.50000001, gpu_allocator_limit_bytes=38*1024**3,
                spectrum_cycles=[60], observer_sample_chunk=1, covariance_spectral_clip=False,
                gap_mode_vectors=True, gap_placeholder=100., gap_tie_tolerance=1e-12,
                eigenmode_basis='active_centered_covariance_eigenvectors')


def extension_config():
    """Ninety NEW trajectories per alpha; existing samples 0--9 stay untouched."""
    config = default_config()
    config.update(sampling_revision='hard_wall_alpha21_nx20_ny30_add90_t60_endpoint_modes_v2',
                  samples_per_case=90, sample_start=10, execution_batch_size=90,
                  combined_samples_per_case=100,
                  previous_sampling_revision=default_config()['sampling_revision'])
    return config


@dataclass(frozen=True)
class Task:
    Ny: int = 30
    alpha_1: float = 1.
    Nx: int = 20
    samples: int = 10
    cycles: int = 60
    root_seed: int = 2026100131
    sample_start: int = 0

    @property
    def name(self):
        return f'Nx{self.Nx:03d}_Ny{self.Ny:03d}_a{self.alpha_1:g}_samples{self.sample_start:03d}-{self.sample_start+self.samples-1:03d}_T{self.cycles:03d}'

    def sample_ids(self, start=0, stop=None):
        return np.arange(self.sample_start+start, self.sample_start+(self.samples if stop is None else stop))

    @property
    def seed(self):
        return int.from_bytes(hashlib.sha256(f'{self.root_seed}|{self.name}'.encode()).digest()[:4], 'little')

    @property
    def active_modes(self):
        return self.Ny*(self.Nx+2)


def tasks(config, lane=None):
    if lane not in (None, 'A', 'B', 'C'):
        raise ValueError('lane must be A, B or C')
    values = [a for i,a in enumerate(config['alpha_values'])
              if lane is None or 'ABC'[i % 3] == lane]
    return [Task(Ny=config['Ny'], Nx=config['Nx'], alpha_1=a,
                 samples=config['samples_per_case'], cycles=config['cycles'],
                 root_seed=config['root_seed'], sample_start=config.get('sample_start', 0))
            for a in sorted(values, key=lambda a: (round(abs(a-2), 12), a))]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(task, config):
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return dict(schema='alpha21_ny30_gap_modes_v1', task=task.name, seed=task.seed,
                config_sha256=hashlib.sha256(canonical.encode()).hexdigest(), configuration=config,
                source_hashes={name: sha(HERE/name) for name in SOURCE_FILES},
                canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit')


def result_paths(output, task, start):
    path = Path(output)/'results'/task.name/f'samples_{task.sample_start+start:03d}-{task.sample_start+min(start+5,task.samples)-1:03d}.npz'
    return path, path.with_suffix('.complete.json')


def checkpoint_paths(output, task):
    path = Path(output)/'checkpoints'/task.name/'checkpoint.npz'
    return path, path.with_suffix('.json')


def pair_verified(path, receipt, expected):
    try:
        row = json.loads(receipt.read_text())
        return (all(compatible_sources(row.get(key), value) if key == 'source_hashes'
                    else row.get(key) == value for key,value in expected.items()) and
                row['filename'] == path.name and row['bytes'] == path.stat().st_size and
                row['sha256'] == sha(path))
    except (OSError, ValueError, KeyError):
        return False


def result_verified(output, task, start, ident):
    path, receipt = result_paths(output, task, start)
    ids = task.sample_ids(start, min(start+5,task.samples)).tolist()
    if not pair_verified(path, receipt, dict(ident, kind='result', sample_indices=ids)):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            validate_products(z, task.cycles)
            active = z['active_indices']
            expected = np.array([mu+2*x+2*task.Nx*y for y in range(task.Ny)
                                 for x in range(task.Nx//4,3*task.Nx//4+1) for mu in range(2)])
            if not np.array_equal(active, expected):
                return False
            coordinates = np.column_stack(((active//2)%task.Nx, active//(2*task.Nx), active%2))
            if not np.array_equal(z['active_coordinates_x_y_orbital'], coordinates):
                return False
            return (np.array_equal(z['sample_indices'], ids) and int(z['T']) == task.cycles and
                    int(z['Nx']) == task.Nx and int(z['Ny']) == task.Ny and
                    z['occupation_spectrum_raw'].shape == (len(ids),task.active_modes) and
                    np.isfinite(z['occupation_spectrum_raw']).all() and
                    float(z['alpha_1']) == task.alpha_1)
    except (OSError, ValueError, KeyError, AssertionError):
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
                   sample_indices=task.sample_ids(), **rng)
    publish_pair(payload, path, receipt, dict(ident, kind='checkpoint', completed_cycle=cycle),
                 Path(scratch)/task.name, compressed=False)
    print(f'[checkpoint] {task.name}: verified cycle {cycle}/{task.cycles}', flush=True)


def load_checkpoint(output, task, ident, segment=10):
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
            payload['G'].shape != (task.samples, 2*task.Nx*task.Ny, 2*task.Nx*task.Ny) or
            payload['G'].dtype != np.complex128 or not np.isfinite(payload['G']).all() or
            not np.array_equal(payload['sample_indices'], task.sample_ids())):
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
                             filling_frac=.5, alpha_1=task.alpha_1, alpha_2=30., trial_orbitals='X',
                             dw_truncation=True, triv_region_local_mode=False,
                             device=device, dtype='complex128', backend='local')
    if tuple(model.DW_loc) != (task.Nx//4, 3*task.Nx//4):
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
    print(f'[task] {task.name}, seed={task.seed}, resume cycle={completed}; walls={task.Nx//4,3*task.Nx//4}', flush=True)
    if checkpoint is None and any(p.exists() for p in checkpoint_paths(output, task)):
        print('[warning] Invalid/partial checkpoint rejected; deterministic restart from cycle zero.', flush=True)
    model = None
    if completed < task.cycles:
        model = build_model(task, device)
        if completed == 0:
            np.random.seed(task.seed); torch.manual_seed(task.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task.seed)
        with tqdm(total=task.cycles, initial=completed, desc=f'Ny={task.Ny} physical cycles', unit='cycle') as bar:
            while completed < task.cycles:
                count = min(config['segment_cycles'], task.cycles-completed)
                begin = time.perf_counter()
                G, rng = run_segment(model, task, G, completed, count, rng, bar)
                duration = time.perf_counter()-begin
                elapsed += duration
                completed += count
                checkpoint_start = time.perf_counter()
                save_checkpoint(output, scratch, task, ident, G, completed, elapsed, rng)
                print(f'[timing] Ny={task.Ny}: {duration/count:.2f}s/cycle; checkpoint {time.perf_counter()-checkpoint_start:.2f}s', flush=True)
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
        payload.update(sample_indices=task.sample_ids(start,stop), active_indices=active,
                       active_coordinates_x_y_orbital=np.column_stack(((active//2)%task.Nx, active//(2*task.Nx), active%2)),
                       Nx=np.array(task.Nx), Ny=np.array(task.Ny), T=np.array(task.cycles),
                       alpha_1=np.array(task.alpha_1), gap_placeholder=np.array(config['gap_placeholder']),
                       walls=np.array([task.Nx//4,3*task.Nx//4]), seed=np.array(task.seed),
                       dynamics_seconds=np.array(elapsed), endpoint_seconds=np.array(endpoint_seconds),
                       configuration_json=np.array(json.dumps(config,sort_keys=True)),
                       source_hashes_json=np.array(json.dumps(ident['source_hashes'],sort_keys=True)))
        publish_pair(payload, path, receipt, dict(ident,kind='result',sample_indices=task.sample_ids(start,stop).tolist()),
                     Path(scratch)/task.name, compressed=True)
        if not result_verified(output, task, start, ident):
            raise OSError('Endpoint result did not verify')
        if outer is not None:
            outer.update(1)
            outer.set_postfix(completed=outer.n, pending=outer.total-outer.n, failed=0)
    cleanup_checkpoint(output, task, ident)
    if str(device).startswith('cuda'):
        peak = torch.cuda.max_memory_reserved()
        print(f'[memory] peak reserved {peak/1024**3:.3f} GiB', flush=True)
        if peak > config['gpu_allocator_limit_bytes']:
            raise RuntimeError('38-GiB allocator ceiling exceeded')
    print(f'[complete] {task.name}: dynamics {elapsed:.1f}s; endpoint {endpoint_seconds:.1f}s', flush=True)


def inventory(config, output, lane=None):
    rows = []
    for task in tasks(config, lane):
        ident = identity(task,config)
        done = sum(result_verified(output,task,s,ident) for s in range(0,task.samples,5))
        checkpoint = load_checkpoint(output,task,ident,config['segment_cycles']) if done < (task.samples+4)//5 else None
        rows.append(dict(task=task.name,Nx=task.Nx,Ny=task.Ny,completed_shards=done,
                         alpha_1=task.alpha_1, recoverable_cycle=None if checkpoint is None else int(checkpoint['completed_cycle'])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', choices=['A','B','C'], required=True)
    parser.add_argument('--config',type=Path,default=HERE/'campaign_config.json')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--scratch-root',type=Path,required=True)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--max-new-execution-batches',type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config not in (default_config(), extension_config()):
        parser.error('Configuration differs from the locked alpha21 Ny30 T=60 contract')
    if args.max_new_execution_batches is not None and args.max_new_execution_batches < 0:
        parser.error('Maximum new batches must be nonnegative')
    rows = inventory(config,args.output_root,args.lane)
    lane_tasks = tasks(config,args.lane)
    total_shards = sum((task.samples+4)//5 for task in lane_tasks)
    done = sum(row['completed_shards'] for row in rows)
    print(json.dumps(dict(config=config,source_root=str(HERE),source_hashes=identity(tasks(config)[0],config)['source_hashes'],
                          output_root=str(args.output_root),scratch_root=str(args.scratch_root),
                          canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit',
                          lane=args.lane,trajectories=sum(t.samples for t in lane_tasks),
                          execution_batches=len(lane_tasks),total_shards=total_shards,completed=done,pending=total_shards-done,
                          inventory=rows),indent=2),flush=True)
    if args.report_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. In Colab select Runtime > Change runtime type > A100 GPU, reconnect, and rerun from the top. If already selected, check torch.cuda.is_available() and nvidia-smi.')
    props = torch.cuda.get_device_properties(torch.device(config['device']))
    if 'A100' not in props.name or props.total_memory < 38*1024**3:
        raise RuntimeError(f'Expected A100 40-GB-class GPU; found {props.name}')
    torch.cuda.set_per_process_memory_fraction(config['gpu_allocator_limit_bytes']/props.total_memory)
    print(f'[device] {props.name}; total {props.total_memory/1e9:.2f} GB; allocator limit 38 GiB',flush=True)
    for path in (args.output_root,args.scratch_root):
        path.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(path).free < 8*1024**3:
            raise OSError(f'Need 8 GiB free for temporary and rolling checkpoint copies: {path}')
    launched = 0
    session_start = time.perf_counter()
    with tqdm(total=total_shards,initial=done,desc='durable endpoint shards',unit='shard') as bar:
        bar.set_postfix(completed=done, skipped=done, pending=total_shards-done, failed=0)
        for task,row in zip(lane_tasks,rows):
            if row['completed_shards'] == (task.samples+4)//5:
                if any(p.exists() for p in checkpoint_paths(args.output_root, task)):
                    cleanup_checkpoint(args.output_root, task, identity(task, config))
                continue
            if args.max_new_execution_batches is not None and launched >= args.max_new_execution_batches:
                break
            torch.cuda.reset_peak_memory_stats()
            try:
                run_task(task,config,args.output_root,args.scratch_root,bar)
            except Exception:
                bar.set_postfix(completed=bar.n, skipped=done, pending=total_shards-bar.n, failed=1)
                raise
            bar.set_postfix(completed=bar.n, skipped=done, pending=total_shards-bar.n, failed=0)
            launched += 1
            torch.cuda.empty_cache()
            remaining = sum(r['completed_shards'] < (t.samples+4)//5 for t,r in zip(lane_tasks,rows))-launched
            print(f'[ETA] lane {args.lane}: {max(0,remaining)*(time.perf_counter()-session_start)/launched/3600:.2f} hours remaining (measured session average)', flush=True)
    print('[summary] '+json.dumps(inventory(config,args.output_root,args.lane)),flush=True)


if __name__ == '__main__':
    main()
