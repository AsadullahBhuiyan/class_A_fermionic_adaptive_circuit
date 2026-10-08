"""Deterministic measured-cost partitions and two-worker schedules."""
import hashlib
import itertools
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def partitions(total, candidates):
    candidates = sorted(set(candidates), reverse=True)
    def visit(remaining, start, prefix):
        if remaining == 0:
            yield prefix
        for index in range(start, len(candidates)):
            size = candidates[index]
            if size <= remaining:
                yield from visit(remaining - size, index, prefix + [size])
    return list(visit(total, 0, []))


def tasks_for(config, ny, partition):
    if sum(partition) != config['samples'] or any(n % config['result_shard_size'] for n in partition):
        raise ValueError('Frozen partitions must cover all samples in whole result shards')
    rows, first = [], 0
    for count in partition:
        stop = first + count
        name = f'Ny{ny:03d}_samples{first:03d}-{stop-1:03d}'
        seed = int(digest([config['root_seed'], config['revision'], name])[:8], 16)
        rows.append(dict(task_id=name, ny=ny, first=first, stop=stop, samples=count,
                         cycles=config['cycles_multiplier'] * ny, seed=seed))
        first = stop
    return rows


def freeze_schedule(config, benchmark_rows, source_sha256):
    by_size, all_tasks = {}, []
    for ny in config['Ny_values']:
        benchmark = benchmark_rows[ny]
        eligible = {r['batch_size']: r for r in benchmark['dynamics'] if r['accepted']}
        choices = []
        costs = {}
        for count, trial in eligible.items():
            endpoint = sum(r['seconds_per_pair'] * count * ny for r in benchmark['endpoint'].values())
            costs[count] = (trial['initialization_seconds'] + config['cycles_multiplier'] * ny * trial['seconds_per_cycle']
                + math.ceil(config['cycles_multiplier'] * ny / config['checkpoint_cycles']) * trial['checkpoint_seconds']
                + endpoint + (ny // 2 + 1) * trial['endpoint_publication_seconds'])
        for partition in partitions(config['samples'], eligible):
            choices.append((sum(costs[n] for n in partition), len(partition), partition))
        if not choices:
            raise RuntimeError(f'No safe measured partition covers Ny={ny}')
        seconds, _, partition = min(choices)
        tasks = tasks_for(config, ny, partition)
        for task in tasks:
            task['projected_seconds'] = costs[task['samples']]
            task['matrix_batch_by_ay'] = {str(ay): row['matrix_batch'] for ay, row in benchmark['endpoint'].items()}
        by_size[ny] = dict(ny=ny, partition=partition, projected_seconds=seconds)
        all_tasks.extend(tasks)
    sizes = config['Ny_values']
    whole_choices = []
    for mask in itertools.product((0, 1), repeat=len(sizes) - 1):
        assignment = [0] + list(mask)
        loads = [sum(by_size[ny]['projected_seconds'] for ny, worker in zip(sizes, assignment) if worker == w) for w in (0, 1)]
        whole_choices.append((max(loads), assignment, loads))
    whole_time, assignment, whole_loads = min(whole_choices)
    whole_workers = [[], []]
    for ny, worker in sorted(zip(sizes, assignment), key=lambda p: (-by_size[p[0]]['projected_seconds'], p[0])):
        whole_workers[worker].extend(t for t in all_tasks if t['ny'] == ny)
    batch_workers, batch_loads = [[], []], [0., 0.]
    for task in sorted(all_tasks, key=lambda t: (-t['projected_seconds'], t['task_id'])):
        worker = min((0, 1), key=lambda w: (batch_loads[w], w))
        batch_workers[worker].append(task)
        batch_loads[worker] += task['projected_seconds']
    use_whole = whole_time <= max(batch_loads)
    selected = dict(mode='whole_size' if use_whole else 'execution_batches',
        workers=whole_workers if use_whole else batch_workers,
        predicted_worker_seconds=whole_loads if use_whole else batch_loads,
        alternatives=dict(whole_size_seconds=whole_time, execution_batches_seconds=max(batch_loads)),
        sizes={str(ny): row for ny, row in by_size.items()}, config_sha256=digest(config), source_sha256=source_sha256)
    selected['schedule_sha256'] = digest(selected)
    return selected
