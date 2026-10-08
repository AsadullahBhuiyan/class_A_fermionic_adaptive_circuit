"""Measure isolated dynamics and endpoint candidates, then freeze two workers."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from endpoint import CONTOUR, ENTROPY, empty_endpoint, solve_pairs
from run_campaign import (HERE, advance, gpu_contract, identity, model_for, read_config,
                          source_identity)
from scheduling import digest, freeze_schedule, tasks_for
from storage import atomic_json, capture_rng, publish_npz


def sync():
    torch.cuda.synchronize()


def memory_result(config, free_before, extra=0):
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.max_memory_reserved()
    free_bound = min(torch.cuda.mem_get_info()[0], free_before-reserved) - extra
    accepted = (reserved + extra <= config['allocator_fraction'] * total
                and free_bound >= config['minimum_free_gib'] * 2**30)
    return dict(accepted=bool(accepted), peak_reserved_bytes=reserved,
                minimum_free_bytes=max(0, free_bound), device_bytes=total)


def dynamics_trial(root, config, ny, count):
    free_before, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    np.random.seed(config['root_seed'] + ny + count)
    torch.manual_seed(config['root_seed'] + ny + count)
    started = time.perf_counter()
    model = model_for(config, ny, 'cuda')
    native = advance(model, config, count, 0)
    sync()
    initialization = time.perf_counter() - started
    native = advance(model, config, count, 1, native)
    sync()
    started = time.perf_counter()
    native = advance(model, config, count, 2, native)
    sync()
    per_cycle = (time.perf_counter() - started) / 2
    task = dict(task_id=f'benchmark_Ny{ny}_{count}', ny=ny, samples=count, first=0, stop=count,
                cycles=2*ny, seed=config['root_seed'])
    checkpoint = Path(root)/'benchmarks'/f'Ny{ny:03d}'/f'trial_checkpoint_{count}.npz'
    scratch = Path(os.environ.get('SLURM_TMPDIR', '/tmp'))/'classA-endpoint-benchmark'/str(ny)/str(count)
    started = time.perf_counter()
    publish_npz(checkpoint, dict(frame=native['frame'], ranks=native['ranks'], **capture_rng()),
                identity(config, task), scratch)
    checkpoint_seconds = time.perf_counter()-started
    checkpoint.unlink()
    checkpoint.with_suffix('.json').unlink()
    started = time.perf_counter()
    publish_npz(checkpoint, dict(empty_endpoint(count, config['Nx'], ny), seen_widths=np.ones(ny//2+1, dtype=bool)),
                identity(config, task), scratch)
    endpoint_publication = time.perf_counter()-started
    checkpoint.unlink()
    checkpoint.with_suffix('.json').unlink()
    return dict(batch_size=count, initialization_seconds=initialization, seconds_per_cycle=per_cycle,
                checkpoint_seconds=checkpoint_seconds, endpoint_publication_seconds=endpoint_publication,
                **memory_result(config, free_before))


def endpoint_trial(config, ny, matrix_batch):
    count = math.ceil(matrix_batch / ny)
    torch.manual_seed(config['root_seed'] + ny)
    np.random.seed(config['root_seed'] + ny)
    free_before, _ = torch.cuda.mem_get_info()
    model = model_for(config, ny, 'cuda')
    native = advance(model, config, count, 3)
    frame = torch.as_tensor(native['frame'], device='cuda')
    pairs = torch.arange(matrix_batch, device='cuda')
    samples, origins = pairs // ny, pairs % ny
    rows = {}
    # Reserve the worst additional production-frame capacity when evaluating
    # candidate memory. The benchmark's shorter frame batch must not hide it.
    extra_frame_bytes = max(0, config['samples']-count) * (2*config['Nx']*ny)**2 * 16
    for ay in tqdm(range(1, ny//2+1), desc=f'endpoint Ny{ny} matrices={matrix_batch}', unit='width'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = dict(matrix_batch=matrix_batch, accepted=False)
        try:
            with torch.inference_mode():
                values, _, _ = solve_pairs(frame, samples, origins, nx=config['Nx'], ny=ny, ay=ay,
                                           tolerance=config['occupation_tolerance'])
                sync()
                del values
                started = time.perf_counter()
                values, _, _ = solve_pairs(frame, samples, origins, nx=config['Nx'], ny=ny, ay=ay,
                                           tolerance=config['occupation_tolerance'])
                sync()
                seconds = time.perf_counter()-started
                closure = float(abs(values[CONTOUR].sum((1, 2))-values[ENTROPY]).max())
                if closure > config['closure_tolerance']:
                    raise FloatingPointError(f'Benchmark contour closure failed: {closure}')
                row.update(seconds_per_pair=seconds/matrix_batch, closure_error=closure,
                           **memory_result(config, free_before, extra=extra_frame_bytes))
                del values
        except torch.cuda.OutOfMemoryError:
            row.update(reason='CUDA out of memory')
            torch.cuda.empty_cache()
        rows[str(ay)] = row
    return dict(matrix_batch=matrix_batch, widths=rows)


def run_child(root, config, ny, kind, count):
    directory = Path(root)/'benchmarks'/f'Ny{ny:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/f'{kind}_{count}.json'
    expected = dict(config_sha256=digest(config), source_sha256=source_identity(), ny=ny, kind=kind, count=count)
    if output.exists():
        value = json.loads(output.read_text())
        if all(value.get(k) == v for k, v in expected.items()):
            return value['measurement']
        raise ValueError(f'Mismatched benchmark receipt: {output}')
    command = [sys.executable, '-u', str(HERE/'benchmark.py'), '--output-root', str(root),
               '--ny', str(ny), '--trial', kind, '--count', str(count)]
    subprocess.run(command, check=True)
    value = json.loads(output.read_text())
    if not all(value.get(k) == v for k, v in expected.items()):
        raise ValueError('Benchmark child identity mismatch')
    return value['measurement']


def benchmark_size(root, config, ny):
    expected = dict(config_sha256=digest(config), source_sha256=source_identity(), ny=ny)
    complete = Path(root)/'benchmarks'/f'Ny{ny:03d}'/'complete.json'
    if complete.exists():
        value = json.loads(complete.read_text())
        if all(value.get(k) == v for k, v in expected.items()):
            print('[BENCHMARK verified]', ny, flush=True)
            return
        raise ValueError('Mismatched completed benchmark')
    dynamics = [run_child(root, config, ny, 'dynamics', count)
                for count in tqdm(config['dynamics_batch_candidates'], desc=f'Ny{ny} dynamics candidates')]
    endpoints = [run_child(root, config, ny, 'endpoint', count)
                 for count in tqdm(config['matrix_batch_candidates'], desc=f'Ny{ny} endpoint candidates')]
    selected = {}
    for ay in range(1, ny//2+1):
        eligible = [row['widths'][str(ay)] for row in endpoints if row['widths'][str(ay)]['accepted']]
        if not eligible:
            raise RuntimeError(f'No safe endpoint matrix batch for Ny={ny}, Ay={ay}')
        selected[str(ay)] = min(eligible, key=lambda r: (r['seconds_per_pair'], r['matrix_batch']))
    atomic_json(complete, dict(expected, dynamics=dynamics, endpoint=selected, gpu=gpu_contract(config)))
    print('[BENCHMARK DONE]', ny, flush=True)


def freeze(root, config):
    rows = {}
    for ny in config['Ny_values']:
        row = json.loads((Path(root)/'benchmarks'/f'Ny{ny:03d}'/'complete.json').read_text())
        if row['config_sha256'] != digest(config) or row['source_sha256'] != source_identity():
            raise ValueError('Benchmark identities differ')
        rows[ny] = row
    validation = json.loads((Path(root)/'validation/gpu_complete.json').read_text())
    if not validation['passed'] or validation['source_sha256'] != source_identity() or validation['config_sha256'] != digest(config):
        raise ValueError('GPU validation must pass before schedule freeze')
    schedule = freeze_schedule(config, rows, source_identity())
    path = Path(root)/'schedule.json'
    if path.exists():
        if json.loads(path.read_text()) != schedule:
            raise ValueError('Preserving an already frozen schedule')
        return
    atomic_json(path, schedule)
    print('[FROZEN]', json.dumps(schedule, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--ny', type=int)
    parser.add_argument('--trial', choices=('dynamics', 'endpoint'))
    parser.add_argument('--count', type=int)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    config = read_config(args.output_root)
    if args.freeze:
        freeze(args.output_root, config)
        return
    gpu_contract(config)
    ny = args.ny or config['Ny_values'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    if ny not in config['Ny_values']:
        raise ValueError('Unknown benchmark circumference')
    if args.trial:
        try:
            row = (dynamics_trial(args.output_root, config, ny, args.count) if args.trial == 'dynamics'
                   else endpoint_trial(config, ny, args.count))
        except torch.cuda.OutOfMemoryError:
            if args.trial != 'dynamics':
                raise
            row = dict(batch_size=args.count, accepted=False, reason='CUDA out of memory')
        output = args.output_root/'benchmarks'/f'Ny{ny:03d}'/f'{args.trial}_{args.count}.json'
        atomic_json(output, dict(config_sha256=digest(config), source_sha256=source_identity(), ny=ny,
                               kind=args.trial, count=args.count, measurement=row))
        print('[MEASUREMENT]', args.trial, ny, args.count, json.dumps(row), flush=True)
    else:
        benchmark_size(args.output_root, config, ny)


if __name__ == '__main__':
    main()
