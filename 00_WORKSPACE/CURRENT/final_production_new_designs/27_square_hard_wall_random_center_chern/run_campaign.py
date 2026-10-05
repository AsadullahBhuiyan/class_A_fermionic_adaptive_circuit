"""Single-process A100 campaign: calibrated immutable batches, simple resume."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch
from tqdm import tqdm

from random_center_observer import RandomCenterObserver, center_choices, ensemble_statistics

ROOT = Path(__file__).resolve().parent
REVISION = 'square_hard_wall_random_center_chern_l20-30-40_s100_t40_r0p2_v1'
SOURCE_FILES = ('run_campaign.py', 'random_center_observer.py',
                'src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py')


def default_config():
    return dict(sampling_revision=REVISION, root_seed=2026092801,
                sizes=[20, 30, 40], samples=100, cycles=40,
                alpha_1=1., alpha_2=30., nshell=1, DW=True,
                dw_truncation=True, triv_region_local_mode=False,
                meas_slab_only=True, perfect_correction=True, postselect=False,
                postselect_probability=0., sequence='raster_y',
                init_mode='default', filling_frac=.5, n_a=.5, trial_orbitals='X',
                boundary_conditions='periodic', dtype='complex128', backend='local',
                state_representation='physical_frame', frame_reorthonormalize_interval=1,
                centers_per_cycle=10, radius_fraction=.2, center_chunk_size=10,
                calibration_batches=[5, 10, 25, 50, 100], fallback_batches=[2, 1],
                warmup_cycles=1, timed_cycles=5, memory_limit_gib=32.,
                forecast_margin=1.25, max_forecast_minutes=45.,
                stop_after_batch_seconds=3600.)


def interfaces(nx):
    """Canonical integer slab boundaries, including the non-multiple-of-four L=30."""
    return (nx // 2 - max(1, nx // 4), nx // 2 + max(1, nx // 4))


def validate_config(config):
    # A new scientific contract belongs to a different explicitly versioned bundle.
    if set(config) != set(default_config()):
        raise ValueError('configuration fields differ from this campaign contract')
    editable = {'center_chunk_size'}
    if any(config[k] != v for k, v in default_config().items() if k not in editable):
        raise ValueError('this revision fixes the scientific and calibration contract; use a new revision for changes')
    if not isinstance(config['center_chunk_size'], int) or not 1 <= config['center_chunk_size'] <= 10:
        raise ValueError('center_chunk_size must be an integer in [1,10]')


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=h.hexdigest())


def identity(config):
    return dict(config_sha256=hashlib.sha256(json_bytes(config)).hexdigest(),
                sources={name: digest(ROOT / name)['sha256'] for name in SOURCE_FILES})


def task_table(config, batches):
    tasks = []
    for ny in config['sizes']:
        size = int(batches[str(ny)])
        if size not in config['calibration_batches'] + config['fallback_batches']:
            raise ValueError('batch size was not a calibration candidate')
        for start in range(0, config['samples'], size):
            stop = min(start + size, config['samples'])
            seed = int(np.random.SeedSequence(
                [config['root_seed'], 2702, ny, ny, start, stop]
            ).generate_state(1, dtype=np.uint32)[0])
            tasks.append(dict(id=f'nx{ny}_ny{ny}_s{start:03d}-{stop-1:03d}',
                              nx=ny, ny=ny, sample_ids=list(range(start, stop)), seed=seed))
    return tasks


def atomic_copy(source, destination):
    """DriveFS-only durability, NOT a promise of cloud/server visibility."""
    source, destination = Path(source), Path(destination)
    expected = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp',
                                dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise IOError(f'DriveFS readback failed: {temporary}')
        os.replace(temporary, destination)
        if digest(destination) != expected:
            raise IOError(f'final DriveFS readback failed: {destination}')
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def publish_json(value, destination, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        path = Path(tmp) / 'value.json'
        path.write_bytes(json_bytes(value))
        atomic_copy(path, destination)


def validate_arrays(arrays, task, config):
    b, ny = len(task['sample_ids']), task['ny']
    t = config['cycles'] + 1
    k = config['centers_per_cycle']
    if not np.array_equal(arrays['sample_ids'], task['sample_ids']):
        raise ValueError('sample IDs mismatch')
    if not np.array_equal(arrays['cycles'], np.arange(t)):
        raise ValueError('cycle coverage mismatch')
    centers, chern = arrays['centers_y'], arrays['real_space_chern']
    if centers.shape != (b, t, k) or chern.shape != (b, t, k):
        raise ValueError('center/observation shapes mismatch')
    if centers.dtype.kind not in 'iu' or np.any((centers < 0) | (centers >= ny)):
        raise ValueError('centers must be in-range integers')
    if np.any(np.diff(np.sort(centers, axis=2), axis=2) == 0):
        raise ValueError('repeated centers')
    for cycle in range(t):
        expected = center_choices(config['root_seed'], task['nx'], ny,
                                  task['sample_ids'], cycle, k)
        if not np.array_equal(centers[:, cycle], expected):
            raise ValueError('center RNG identity mismatch')
    average = arrays['center_average']
    if average.shape != (b, t) or not np.isfinite(chern).all():
        raise ValueError('invalid Chern arrays')
    if not np.array_equal(average, chern.mean(axis=2)):
        raise ValueError('incorrect trajectory-first center mean')
    frame, ranks = arrays['final_frame'], arrays['final_ranks']
    charge = arrays['global_charge']
    if frame.ndim != 3 or frame.shape[:2] != (b, 2*task['nx']*ny) or frame.dtype != np.complex128:
        raise ValueError('invalid final frame shape/dtype')
    if ranks.shape != (b,) or ranks.dtype.kind not in 'iu' or np.any((ranks < 0) | (ranks > frame.shape[2])):
        raise ValueError('invalid final ranks')
    if charge.shape != (b, t) or charge.dtype.kind not in 'iu' or np.any((charge < 0) | (charge > frame.shape[1])):
        raise ValueError('invalid integer charges')
    if not np.array_equal(charge[:, -1], ranks):
        raise ValueError('endpoint charge/rank mismatch')
    for row, rank in zip(frame, ranks):
        if not np.isfinite(row).all() or np.any(row[:, rank:] != 0):
            raise ValueError('nonfinite frame or nonzero padding')
        # A cheap independent trace/charge check; engine enforces orthonormality.
        if not np.isclose(np.vdot(row, row).real, rank, atol=1e-7, rtol=0):
            raise ValueError('frame norm/charge mismatch')


def verified_result(output, task, ident, config, *, load=False):
    path = Path(output) / (task['id'] + '.npz')
    receipt_path = path.with_suffix('.json')
    if not path.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
        if (receipt['task'] != task or receipt['identity'] != ident
                or receipt['result'] != path.name or receipt['file'] != digest(path)):
            return None
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata_json']))
            if metadata != dict(task=task, identity=ident, config=config,
                                entry_point='classA_U1FGTN_gpu.run_markov_circuit'):
                return None
            validate_arrays(saved, task, config)
            return ({key: saved[key] for key in ('sample_ids', 'center_average', 'cycles')}
                    if load else receipt)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None


def publish_result(arrays, task, config, ident, output, scratch, elapsed):
    validate_arrays(arrays, task, config)
    metadata = dict(task=task, identity=ident, config=config,
                    entry_point='classA_U1FGTN_gpu.run_markov_circuit')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        local = Path(tmp) / (task['id'] + '.npz')
        np.savez_compressed(local, **arrays, metadata_json=np.array(json.dumps(metadata)))
        output_path = output / local.name
        checksum = atomic_copy(local, output_path)
        receipt = dict(task=task, identity=ident, result=local.name, file=checksum,
                       elapsed_seconds=elapsed, durability='DriveFS_readback_only')
        publish_json(receipt, output_path.with_suffix('.json'), scratch)
    if not verified_result(output, task, ident, config):
        raise IOError('published pair failed verification')


def require_a100():
    if not torch.cuda.is_available():
        raise RuntimeError('Production and calibration require an A100 40-GB-class GPU')
    device = torch.cuda.get_device_properties(0)
    if 'A100' not in device.name or device.total_memory < 35 * 1024**3:
        raise RuntimeError(f'Expected A100 40-GB-class GPU, got {device.name}')
    print(f'[device] {device.name}; {device.total_memory/1024**3:.2f} GiB; complex128', flush=True)


def run_dynamics(config, task, cycles, *, calibration=False):
    sys.path.insert(0, str(ROOT / 'src'))
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu
    seed = task['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    model = classA_U1FGTN_gpu(
        Nx=task['nx'], Ny=task['ny'], DW=config['DW'], nshell=config['nshell'],
        filling_frac=config['filling_frac'], alpha_1=config['alpha_1'],
        alpha_2=config['alpha_2'], trial_orbitals=config['trial_orbitals'],
        dw_truncation=config['dw_truncation'],
        triv_region_local_mode=config['triv_region_local_mode'],
        device='cuda', dtype=config['dtype'], backend=config['backend'])
    if tuple(model.DW_loc) != interfaces(task['nx']) or model.dtype != torch.complex128:
        raise RuntimeError('hard-wall geometry/dtype mismatch')
    observer = RandomCenterObserver(task['nx'], task['ny'], cycles, task['sample_ids'],
                                    config['root_seed'] + (1 if calibration else 0),
                                    radius=config['radius_fraction'] * task['nx'],
                                    count=config['centers_per_cycle'],
                                    chunk_size=config['center_chunk_size'])
    b = len(task['sample_ids'])
    result = model.run_markov_circuit(
        G_history=False, progress=True, cycles=cycles, samples=b, batch_size=b,
        init_mode=config['init_mode'], save=False, n_a=config['n_a'],
        postselect=config['postselect'], postselect_probability=config['postselect_probability'],
        perfect_correction=config['perfect_correction'], sequence=config['sequence'],
        meas_slab_only=config['meas_slab_only'], return_data=True,
        state_representation=config['state_representation'],
        native_cycle_observer=observer.capture, track_choi=False,
        return_native_state=True, require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=config['frame_reorthonormalize_interval'])
    torch.cuda.synchronize()
    if (result['state_representation_resolved'] != 'physical_frame'
            or result['covariance_materialization_count'] != 0
            or not result['exterior_preparation_performed'] or result['samples'] != b):
        raise RuntimeError('canonical engine contract mismatch')
    arrays = observer.arrays()
    native = result['native_final']
    if calibration:
        arrays = None
    else:
        # Canonical return_native_state exports NumPy arrays (snapshot(cpu=True)).
        frame = np.array(native['frame'], dtype=np.complex128, copy=True)
        ranks = np.asarray(native['ranks'], dtype=np.int64)
        # Store only the largest actual rank; remove unused padded capacity.
        frame = frame[:, :, :int(ranks.max())].copy()
        for row, rank in zip(frame, ranks):
            row[:, rank:] = 0
        arrays.update(final_frame=frame, final_ranks=ranks)
    metrics = dict(elapsed_seconds=time.perf_counter()-started,
                   setup_seconds=observer.cycle_times[0]-started,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    if calibration:
        warmup = config['warmup_cycles']
        metrics['seconds_per_cycle'] = ((observer.cycle_times[cycles] - observer.cycle_times[warmup])
                                       / config['timed_cycles'])
    return arrays, metrics


def select_candidate(rows, config):
    accepted = [r for r in rows if r.get('status') == 'ok'
                and r['peak_reserved_gib'] < config['memory_limit_gib']
                and r['forecast_seconds'] < config['max_forecast_minutes'] * 60]
    if not accepted:
        return None
    return max(accepted, key=lambda r: r['batch'] / r['seconds_per_cycle'])


def calibrate(config):
    sizes, all_rows = {}, {}
    for ny in config['sizes']:
        rows = []
        for candidates in (config['calibration_batches'], config['fallback_batches']):
            for b in candidates:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                seed = int(np.random.SeedSequence([config['root_seed'], 2799, ny, b])
                           .generate_state(1, dtype=np.uint32)[0])
                task = dict(nx=ny, ny=ny, sample_ids=list(range(b)), seed=seed)
                print(f'[calibration] Ny={ny}, B={b}: 1 warm-up + 5 timed cycles', flush=True)
                try:
                    _, metrics = run_dynamics(config, task, config['warmup_cycles'] + config['timed_cycles'],
                                              calibration=True)
                    row = dict(batch=b, seed=seed, status='ok', **metrics)
                    row['forecast_seconds'] = config['forecast_margin'] * (
                        metrics['setup_seconds'] + config['cycles'] * metrics['seconds_per_cycle'])
                except torch.cuda.OutOfMemoryError:
                    row = dict(batch=b, seed=seed, status='out_of_memory')
                rows.append(row)
                print('[measured] ' + json.dumps(row), flush=True)
                # Ascending main candidates: stop growing a batch once a measured
                # resource limit is exceeded. No need to provoke still larger OOMs.
                if candidates is config['calibration_batches'] and select_candidate([row], config) is None:
                    break
            selected = select_candidate(rows, config)
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f'Ny={ny}: no batch met memory/time limits; production NOT launched')
        sizes[str(ny)] = selected['batch']
        all_rows[str(ny)] = rows
        print(f'[selected] Ny={ny}: B={selected["batch"]}; {selected["forecast_seconds"]/60:.1f} min/batch (25% margin)', flush=True)
    return sizes, all_rows


def validate_plan(plan, config, ident):
    if plan['config'] != config or plan['identity'] != ident:
        raise RuntimeError('existing execution plan has different config/source identity; do not overwrite it')
    if plan['tasks'] != task_table(config, plan['batch_sizes']):
        raise RuntimeError('execution plan task/seed mismatch')
    for ny, b in plan['batch_sizes'].items():
        selected = select_candidate(plan['calibration'][ny], config)
        if selected is None or selected['batch'] != b:
            raise RuntimeError('execution plan lacks a qualifying measured calibration')


def preview(output):
    """Read-only notebook preview, rebuilding statistics from verified pairs."""
    output = Path(output)
    path = output / 'execution_plan.json'
    if not path.is_file():
        return {}
    plan = json.loads(path.read_text())
    config, ident = plan['config'], plan['identity']
    validate_plan(plan, config, ident)
    products = {}
    for ny in config['sizes']:
        rows = [verified_result(output, task, ident, config, load=True)
                for task in plan['tasks'] if task['ny'] == ny]
        rows = [r for r in rows if r is not None]
        if rows:
            values = np.concatenate([r['center_average'] for r in rows])
            mean, sem = ensemble_statistics(values)
            products[ny] = dict(cycles=rows[0]['cycles'], mean=mean, sem=sem, samples=len(values))
    return products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'campaign_config.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, default=Path('/content/square_random_center_chern_scratch'))
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--max-new-batches', type=int)
    args = parser.parse_args(argv)
    if args.max_new_batches is not None and args.max_new_batches < 0:
        parser.error('--max-new-batches must be nonnegative')
    config = json.loads(args.config.read_text())
    validate_config(config)
    ident = identity(config)
    output, scratch = args.output_root, args.scratch_root
    print(json.dumps(dict(config=config, identity=ident, source_root=str(ROOT),
                          output=str(output), scratch=str(scratch), trajectories=300,
                          individual_chern_measurements=123000,
                          geometry=[dict(nx=l, ny=l, interfaces=interfaces(l), radius=.2*l)
                                    for l in config['sizes']]), indent=2), flush=True)
    plan_path = output/'execution_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        validate_plan(plan, config, ident)
    elif args.report_only:
        print('[inventory] not calibrated; 300 trajectories pending; no production launched', flush=True)
        return
    else:
        if output.exists() and any(output.glob('nx*_ny*_s*.npz')):
            raise RuntimeError('results exist without execution_plan.json; restore the original plan')
        require_a100()
        output.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        # Reserve room for local uncompressed final frames + compressed copy.
        if shutil.disk_usage(scratch).free < 12*1024**3 or shutil.disk_usage(output).free < 12*1024**3:
            raise RuntimeError('need at least 12 GiB free on scratch and mounted Drive')
        sizes, measurements = calibrate(config)
        plan = dict(config=config, identity=ident, batch_sizes=sizes,
                    calibration=measurements, tasks=task_table(config, sizes))
        validate_plan(plan, config, ident)
        publish_json(plan, plan_path, scratch)
    tasks = plan['tasks']
    valid = {t['id']: verified_result(output, t, ident, config) for t in tasks}
    complete = sum(v is not None for v in valid.values())
    print(f'[inventory] {complete}/{len(tasks)} verified batches; {len(tasks)-complete} pending', flush=True)
    forecast = sum(next(r['forecast_seconds'] for r in plan['calibration'][str(t['ny'])]
                        if r['batch'] == plan['batch_sizes'][str(t['ny'])] and r['status']=='ok')
                   * len(t['sample_ids'])/plan['batch_sizes'][str(t['ny'])]
                   for t in tasks if valid[t['id']] is None)
    print(f'[forecast] remaining dynamics/setup approximately {forecast/3600:.2f} A100 hours; excludes compression/Drive transfer', flush=True)
    if args.report_only or complete == len(tasks):
        return
    # Persist the stop through restarts, including a crash just after publication.
    if any(v is not None and v['elapsed_seconds'] > config['stop_after_batch_seconds'] for v in valid.values()):
        raise RuntimeError('a saved batch exceeded one hour; revise execution sizes before continuing (do not edit the frozen plan)')
    require_a100()
    scratch.mkdir(parents=True, exist_ok=True)
    completed_now, skipped, failed = 0, 0, 0
    with tqdm(tasks, desc='hard-wall Chern batches', unit='batch', dynamic_ncols=True) as bar:
        for task in bar:
            bar.set_postfix(completed=complete, skipped=skipped, pending=len(tasks)-complete, failed=failed)
            if valid[task['id']] is not None:
                skipped += 1
                continue
            if args.max_new_batches is not None and completed_now >= args.max_new_batches:
                break
            print(f'[batch] {task["id"]}; seed={task["seed"]}', flush=True)
            if shutil.disk_usage(scratch).free < 12*1024**3 or shutil.disk_usage(output).free < 12*1024**3:
                raise RuntimeError('insufficient free scratch/Drive space; batch not started')
            started = time.perf_counter()
            try:
                torch.cuda.reset_peak_memory_stats()
                arrays, metrics = run_dynamics(config, task, config['cycles'])
                publish_result(arrays, task, config, ident, output, scratch, time.perf_counter()-started)
                elapsed = time.perf_counter()-started
                # Include compression/transfer time in the persistent one-hour stop.
                receipt_path = output/(task['id']+'.json')
                receipt = json.loads(receipt_path.read_text())
                receipt.update(elapsed_seconds=elapsed, metrics=metrics)
                publish_json(receipt, receipt_path, scratch)
                del arrays
                completed_now += 1
                complete += 1
                print(f'[saved] {task["id"]}: {elapsed/60:.1f} min; peak reserved {metrics["peak_reserved_gib"]:.2f} GiB', flush=True)
                if elapsed > config['stop_after_batch_seconds']:
                    raise RuntimeError('batch saved, but exceeded one hour; queue stopped before next batch')
            except Exception:
                failed += 1
                bar.set_postfix(completed=complete, skipped=skipped, pending=len(tasks)-complete, failed=failed)
                raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    print(f'[finished session] {complete}/{len(tasks)} verified batches; {len(tasks)-complete} pending', flush=True)


if __name__ == '__main__':
    main()
