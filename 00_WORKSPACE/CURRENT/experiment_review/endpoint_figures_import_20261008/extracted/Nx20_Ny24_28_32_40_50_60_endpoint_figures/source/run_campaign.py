"""Frozen native-frame dynamics and endpoint-only scientific products."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from endpoint import CONTOUR, ENTROPY, VARIANCE, KEYS, averaged_width, empty_endpoint, validate_endpoint
from scheduling import digest
from storage import atomic_json, capture_rng, load_checkpoint, publish_npz, restore_rng, sha, verified_pair

HERE = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get('CLASSA_ENGINE_DIR', HERE/'engine' if (HERE/'engine').is_dir() else HERE.parents[3]/'src/fgtn'))
sys.path.insert(0, str(ENGINE))
from classA_U1FGTN_gpu import classA_U1FGTN_gpu
ENTRY_POINT = 'classA_U1FGTN_gpu.run_markov_circuit'


@lru_cache(maxsize=1)
def source_identity():
    files = {p.name: sha(p) for pattern in ('*.py', '*.sbatch') for p in sorted(HERE.glob(pattern))}
    files.update({f'engine/{name}': sha(ENGINE/name) for name in ('classA_U1FGTN_gpu.py', 'occupied_frame_gpu.py')})
    return digest(files)


def read_config(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    config = json.loads((root/'source/campaign_config.json').read_text())
    if manifest['config_sha256'] != digest(config) or manifest['source_sha256'] != source_identity():
        raise ValueError('Frozen configuration/source identity mismatch')
    return config


def read_schedule(root, config):
    schedule = json.loads((Path(root)/'schedule.json').read_text())
    checksum = schedule.pop('schedule_sha256')
    if checksum != digest(schedule):
        raise ValueError('Frozen schedule checksum mismatch')
    schedule['schedule_sha256'] = checksum
    if schedule['config_sha256'] != digest(config) or schedule['source_sha256'] != source_identity():
        raise ValueError('Frozen schedule identity mismatch')
    tasks = [task for worker in schedule['workers'] for task in worker]
    if len({t['task_id'] for t in tasks}) != len(tasks) or len({t['seed'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task or seed in frozen schedule')
    for ny in config['Ny_values']:
        indices = sorted(i for t in tasks if t['ny'] == ny for i in range(t['first'], t['stop']))
        if indices != list(range(config['samples'])):
            raise ValueError(f'Frozen schedule does not cover Ny={ny} exactly once')
    return schedule


def identity(config, task):
    return dict(revision=config['revision'], config_sha256=digest(config), source_sha256=source_identity(),
        task_id=task['task_id'], seed=task['seed'], Nx=config['Nx'], Ny=task['ny'],
        endpoint_cycle=task['cycles'], global_sample_indices=list(range(task['first'], task['stop'])),
        canonical_entry_point=ENTRY_POINT)


def shard_paths(root, task, shard_size):
    return [Path(root)/'results'/f"Ny{task['ny']:03d}"/f'samples_{first:03d}-{first+shard_size-1:03d}.npz'
            for first in range(task['first'], task['stop'], shard_size)]


def shard_identity(config, task, first):
    row = identity(config, task)
    row['global_sample_indices'] = list(range(first, first + config['result_shard_size']))
    row['schema'] = 'b200_endpoint_entropy_contours_result_v1'
    return row


def verified_complete(root, config, task):
    return all(verified_pair(path, shard_identity(config, task, first))
        for path, first in zip(shard_paths(root, task, config['result_shard_size']),
                               range(task['first'], task['stop'], config['result_shard_size'])))


def model_for(config, ny, device):
    return classA_U1FGTN_gpu(Nx=config['Nx'], Ny=ny, DW=config['DW'], nshell=config['nshell'],
        alpha_1=config['alpha_1'], alpha_2=config['alpha_2'], trial_orbitals=config['trial_orbitals'],
        dw_truncation=config['dw_truncation'], filling_frac=config['filling_frac'],
        device=device, dtype=config['dtype'], backend='local')


def engine_kwargs(config, count):
    return dict(samples=count, batch_size=count, G_history=False, progress=False, save=False,
        save_init=False, return_data=True, init_mode=config['init_mode'], state_representation='physical_frame',
        return_native_state=True, require_no_covariance_materialization=True, frame_reorthonormalize_interval=1,
        perfect_correction=config['perfect_correction'], postselect=config['postselect'], n_a=config['n_a'],
        sequence=config['sequence'], meas_slab_only=config['meas_slab_only'], track_choi=False,
        covariance_spectral_clip=False)


def advance(model, config, count, cycles, native=None, observer=None):
    kwargs = engine_kwargs(config, count)
    if native is not None:
        kwargs.update(frame_init=native['frame'], frame_ranks=native['ranks'], frame_init_prepared=True)
    result = model.run_markov_circuit(**kwargs, cycles=cycles, native_cycle_observer=observer)
    if (result.get('state_representation_resolved') != 'physical_frame'
            or result.get('covariance_materialization_count') != 0 or result.get('choi_tracked')):
        raise RuntimeError('Canonical engine violated native-frame contract')
    if bool(result.get('exterior_preparation_performed')) != (native is None):
        raise RuntimeError('Exterior preparation was repeated or omitted')
    return result['native_final']


def validate_native(native, config, task):
    frame, ranks = native['frame'], native['ranks']
    if (frame.ndim != 3 or frame.shape[:2] != (task['samples'], 2 * config['Nx'] * task['ny'])
            or frame.dtype != np.complex128 or not np.isfinite(frame).all()
            or ranks.shape != (task['samples'],) or ranks.dtype != np.int64
            or np.any(ranks < 0) or np.any(ranks > frame.shape[2])):
        raise ValueError('Invalid native checkpoint frame/ranks')
    for index, rank in enumerate(ranks):
        if np.max(abs(frame[index, :, rank:]), initial=0) > 1e-10:
            raise ValueError('Nonzero padded native-frame columns')


def run_task(root, config, task, *, device='cuda', max_segments=None, max_widths=None,
             segment_cycles=None, retain_checkpoints=False):
    root = Path(root)
    if verified_complete(root, config, task):
        print('[SKIP verified]', task['task_id'], flush=True)
        return True
    expected = identity(config, task)
    directory = root/'checkpoints'/task['task_id']
    scratch = Path(os.environ.get('SLURM_TMPDIR', '/tmp'))/'classA-endpoint'/root.name/task['task_id']
    dynamics_path, endpoint_path = directory/'checkpoint.npz', directory/'endpoint_progress.npz'
    checkpoint = load_checkpoint(dynamics_path, expected)
    model = model_for(config, task['ny'], device)
    wanted_walls = [config['Nx']//4, 3*config['Nx']//4]
    if list(model.DW_loc) != wanted_walls:
        raise ValueError('Engine wall geometry differs from campaign')
    completed, native = 0, None
    if checkpoint is not None:
        completed = int(checkpoint['completed_cycle'])
        if not 0 <= completed <= task['cycles']:
            raise ValueError('Invalid completed cycle')
        native = dict(frame=checkpoint['frame'], ranks=checkpoint['ranks'])
        validate_native(native, config, task)
        rng = checkpoint
    else:
        np.random.seed(task['seed'])
        torch.manual_seed(task['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task['seed'])
        rng = capture_rng()
    atomic_json(root/'metadata'/f"{task['task_id']}.json", dict(expected, task=task, config=config,
        wall_positions=wanted_walls, cycle_zero_semantics='after_born_conditioned_exterior_preparation',
        entropy_units='nats', observable_dtype='float64', origin_average_count=task['ny'],
        contour_coordinate='relative_dy=(y-y0)_mod_Ny', job_id=os.environ.get('SLURM_JOB_ID')))
    print('[TASK]', task['task_id'], 'resume cycle', completed, flush=True)
    segments = 0
    with tqdm(total=task['cycles'], initial=completed, desc=task['task_id'], unit='cycle') as bar:
        while completed < task['cycles']:
            count = min(segment_cycles or config['checkpoint_cycles'], task['cycles'] - completed)
            def observe(cycle, batch_start, batch_count, state, **_):
                if batch_start != 0 or batch_count != task['samples']:
                    raise RuntimeError('Engine split frozen dynamics batch')
                if cycle:
                    bar.update(1)
                if not torch.isfinite(state.frame).all():
                    raise FloatingPointError('Nonfinite native frame')
            restore_rng(rng)
            native = advance(model, config, task['samples'], count, native, observe)
            completed += count
            rng = capture_rng()
            validate_native(native, config, task)
            publish_npz(dynamics_path, dict(frame=native['frame'], ranks=native['ranks'],
                completed_cycle=np.array(completed), **rng), expected, scratch)
            segments += 1
            if max_segments is not None and segments >= max_segments:
                return False
    final_sha = sha(dynamics_path)
    endpoint_identity = dict(expected, final_checkpoint_sha256=final_sha)
    progress = load_checkpoint(endpoint_path, endpoint_identity)
    values = empty_endpoint(task['samples'], config['Nx'], task['ny'])
    seen = np.zeros(task['ny']//2+1, dtype=bool)
    if progress is not None:
        seen = progress['seen_widths']
        if seen.shape != (task['ny']//2+1,) or seen.dtype != np.bool_:
            raise ValueError('Invalid endpoint progress mask')
        values = {key: progress[key] for key in KEYS}
        validate_endpoint(values, nx=config['Nx'], ny=task['ny'], samples=task['samples'],
            widths=np.flatnonzero(seen), closure_tolerance=config['closure_tolerance'])
    frame = torch.as_tensor(native['frame'], dtype=torch.complex128, device=device)
    widths_done = 0
    with tqdm(total=len(seen), initial=int(seen.sum()), desc=f"Ny{task['ny']} endpoint", unit='width') as widths:
        for ay in range(len(seen)):
            if seen[ay]:
                continue
            matrix_batch = task['matrix_batch_by_ay'].get(str(ay), 1)
            with tqdm(total=task['samples']*task['ny'] if ay else 0, desc=f'Ay={ay}', unit='strip', leave=False) as pairs:
                block, diagnostic = averaged_width(frame, nx=config['Nx'], ny=task['ny'], ay=ay,
                    matrix_batch=matrix_batch, tolerance=config['occupation_tolerance'], progress=pairs.update)
            values[CONTOUR][:, ay, :, :ay] = block[CONTOUR]
            for key in (ENTROPY, VARIANCE):
                values[key][:, ay] = block[key]
            seen[ay] = True
            validate_endpoint(values, nx=config['Nx'], ny=task['ny'], samples=task['samples'],
                widths=np.flatnonzero(seen), closure_tolerance=config['closure_tolerance'])
            publish_npz(endpoint_path, dict(values, seen_widths=seen), endpoint_identity, scratch)
            widths.update(1)
            widths_done += 1
            print('[WIDTH durable]', task['task_id'], ay, diagnostic, flush=True)
            if max_widths is not None and widths_done >= max_widths:
                return False
    del frame
    closure = validate_endpoint(values, nx=config['Nx'], ny=task['ny'], samples=task['samples'],
        closure_tolerance=config['closure_tolerance'])
    for first, path in zip(range(task['first'], task['stop'], config['result_shard_size']),
                           shard_paths(root, task, config['result_shard_size'])):
        expected_shard = shard_identity(config, task, first)
        if verified_pair(path, expected_shard):
            continue
        selected = slice(first-task['first'], first-task['first']+config['result_shard_size'])
        payload = {key: value[selected] for key, value in values.items()}
        payload.update(sample_ids=np.arange(first, first+config['result_shard_size']),
            ay_values=np.arange(task['ny']//2+1), valid_dy_count=np.arange(task['ny']//2+1),
            endpoint_cycle=np.array(task['cycles']), origin_average_count=np.array(task['ny']),
            contour_coordinate=np.array('relative_dy=(y-y0)_mod_Ny'), Nx=np.array(config['Nx']), Ny=np.array(task['ny']))
        publish_npz(path, payload, expected_shard, scratch)
    if not verified_complete(root, config, task):
        raise RuntimeError('Task products failed final verification')
    if not retain_checkpoints:
        for path in (endpoint_path.with_suffix('.json'), endpoint_path, dynamics_path.with_suffix('.json'), dynamics_path):
            path.unlink(missing_ok=True)
    print('[COMPLETE]', task['task_id'], 'closure', closure, flush=True)
    return True


def gpu_contract(config):
    if not torch.cuda.is_available() or 'B200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Production requires an allocated B200')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    return dict(name=torch.cuda.get_device_name(0), bytes=torch.cuda.get_device_properties(0).total_memory)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    config = read_config(args.output_root)
    schedule = read_schedule(args.output_root, config)
    if args.report:
        rows = [dict(task_id=t['task_id'], complete=verified_complete(args.output_root, config, t))
                for worker in schedule['workers'] for t in worker]
        print(json.dumps(dict(schedule_mode=schedule['mode'], tasks=rows), indent=2))
        return
    if args.worker is None:
        parser.error('--worker is required for production')
    print('[GPU]', gpu_contract(config), flush=True)
    tasks = schedule['workers'][args.worker]
    for task in tqdm(tasks, desc=f'B200 worker {args.worker}', unit='batch'):
        run_task(args.output_root, config, task)
    atomic_json(args.output_root/f'worker_{args.worker}_complete.json', dict(config_sha256=digest(config),
        source_sha256=source_identity(), schedule_sha256=schedule['schedule_sha256'], task_ids=[t['task_id'] for t in tasks]))
    print('[WORKER DONE]', args.worker, flush=True)


if __name__ == '__main__':
    main()
