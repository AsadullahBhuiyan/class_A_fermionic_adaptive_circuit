"""One deterministic equilibrium task with simple completion-based DriveFS resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ('run_campaign.py', 'density_correlations.py', 'plot_results.py',
                'src/classA_U1FGTN.py', 'src/occupied_frame.py')


def default_config():
    return dict(revision='flattened_imaginary_time_density_nx20_ny30_hard_v1',
                Nx=20, Ny=30, alpha_1=1., alpha_2=30., nshell=1,
                DW=True, dw_truncation=True, trial_orbitals='X', filling_fraction=.5,
                dtype='complex128', occupation_twist=1e-7, tau_min=1e-3,
                tau_max=1e3, tau_points=240, tau_chunk=8, normalization_tolerance=1e-12)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(config):
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return dict(config=config, config_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                source_hashes={p: sha(HERE/p) for p in SOURCE_FILES},
                construction_entry_point='classA_U1FGTN.construct_OW_projectors',
                dynamics_entry_point=None, task='equilibrium_ground_state', seed=None)


def paths(output, ident):
    result = Path(output)/f"density_{ident['config_sha256'][:16]}.npz"
    return result, result.with_suffix('.complete.json')


def verified(output, ident):
    result, receipt = paths(output, ident)
    try:
        record = json.loads(receipt.read_text())
        return (all(record.get(k) == v for k, v in ident.items()) and
                record['result_filename'] == result.name and
                record['result_bytes'] == result.stat().st_size and
                record['result_sha256'] == sha(result))
    except (OSError, ValueError, KeyError):
        return False


def publish_file(source, target):
    """DriveFS readback, not independent cloud/server verification."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.tmp')
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != Path(source).stat().st_size or sha(temporary) != sha(source):
        raise OSError(f'Readback mismatch: {temporary}')
    os.replace(temporary, target)
    if target.stat().st_size != Path(source).stat().st_size or sha(target) != sha(source):
        raise OSError(f'Final readback mismatch: {target}')


def publish(data, diagnostics, config, ident, output, scratch):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    result, receipt = paths(output, ident)
    local_result = scratch/result.name
    np.savez_compressed(local_result, **data, configuration_json=json.dumps(config, sort_keys=True),
                        diagnostics_json=json.dumps(diagnostics, sort_keys=True),
                        source_hashes_json=json.dumps(ident['source_hashes'], sort_keys=True))
    publish_file(local_result, result)
    record = dict(ident, result_filename=result.name, result_bytes=result.stat().st_size,
                  result_sha256=sha(result), diagnostics=diagnostics)
    local_receipt = scratch/receipt.name
    local_receipt.write_text(json.dumps(record, indent=2)+'\n')
    publish_file(local_receipt, receipt)
    if not verified(output, ident):
        raise OSError('Final result/completion verification failed')
    return result


def run(config, output, scratch, report_only=False, threads=2):
    # This first benchmark is locked; explicit new revisions can be added later.
    if config != default_config():
        raise ValueError('This revision has a locked scientific/configuration contract; use default_config().')
    ident = identity(config)
    done = verified(output, ident)
    print(json.dumps(dict(ident, source_root=str(HERE), output_root=str(output),
                          scratch_root=str(scratch), device='CPU', threads=threads,
                          completed=int(done), pending=int(not done),
                          workload='one ground state, 20 x positions, 241 imaginary times'), indent=2), flush=True)
    if report_only:
        return None
    from density_correlations import calculate
    from plot_results import plot_all
    with tqdm(total=1, initial=int(done), desc='equilibrium task', unit='task') as bar:
        if done:
            print('[resume] verified result skipped', flush=True)
            result, _ = paths(output, ident)
        else:
            Path(output).mkdir(parents=True, exist_ok=True)
            Path(scratch).mkdir(parents=True, exist_ok=True)
            for directory in (output, scratch):
                if shutil.disk_usage(directory).free < 256*1024**2:
                    raise OSError(f'Need at least 256 MiB free at {directory}')
            with threadpool_limits(limits=threads):
                data, diagnostics = calculate(config)
            print('[validation] '+json.dumps(diagnostics), flush=True)
            result = publish(data, diagnostics, config, ident, output, scratch)
            bar.update(1)
    plot_start = time.perf_counter()
    local_figures = Path(scratch)/'figures'
    plot_all(result, local_figures)
    for path in local_figures.iterdir():
        if path.is_file():
            publish_file(path, Path(output)/'figures'/path.name)
    print(f'[complete] {result}; plots {time.perf_counter()-plot_start:.2f}s; no sampling uncertainty', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=HERE/'campaign_config.json')
    p.add_argument('--output-root', type=Path, default=HERE/'results')
    p.add_argument('--scratch-root', type=Path, default=HERE/'scratch')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    run(json.loads(args.config.read_text()), args.output_root, args.scratch_root,
        args.report_only, args.threads)
