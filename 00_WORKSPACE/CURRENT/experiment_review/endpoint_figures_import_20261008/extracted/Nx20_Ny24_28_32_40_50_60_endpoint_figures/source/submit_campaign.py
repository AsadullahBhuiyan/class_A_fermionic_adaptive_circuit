"""Freeze and submit validation -> two-GPU benchmarks -> production -> figures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scheduling import digest
from storage import atomic_json, sha

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
DEFAULT_ROOT = Path('/blue/panhaining/ab2398.cornell/classA_runs/b200_endpoint_entropy_contours')


def snapshot(root):
    if root.exists():
        raise FileExistsError(f'Preserving existing campaign: {root}; use --resume')
    source = root/'source'
    (source/'engine').mkdir(parents=True)
    (root/'logs').mkdir()
    for path in HERE.iterdir():
        if path.suffix in ('.py', '.json', '.sbatch', '.md'):
            shutil.copy2(path, source/path.name)
    engine_hashes = {}
    for name in ('classA_U1FGTN_gpu.py', 'occupied_frame_gpu.py'):
        path = REPOSITORY/'src/fgtn'/name
        shutil.copy2(path, source/'engine'/name)
        engine_hashes[f'engine/{name}'] = sha(path)
    files = {p.name: sha(p) for pattern in ('*.py', '*.sbatch') for p in sorted(source.glob(pattern))}
    files.update(engine_hashes)
    config = json.loads((source/'campaign_config.json').read_text())
    manifest = dict(config=config, config_sha256=digest(config), source_sha256=digest(files), source_hashes=files,
        canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit', repository_bundle=str(HERE),
        created_utc=datetime.now(timezone.utc).isoformat(), GPU='B200', maximum_concurrent_GPUs=2,
        initialization='fresh random pure half-filled trajectories, Born-conditioned product exterior',
        output_observables=['endpoint__contour_von_neumann_y0avg', 'endpoint__entropy_von_neumann', 'endpoint__charge_variance'],
        result_shards=120, independent_trajectories=600)
    atomic_json(root/'manifest.json', manifest)
    return config


def submit(root, dry_run=False):
    jobs = []
    def launch(stage, script, dependency=None, array=None):
        command = ['sbatch', '--parsable', f'--job-name=classA-endpoint-{stage}',
            f'--output={root}/logs/{stage}_%A_%a.out', f'--error={root}/logs/{stage}_%A_%a.err']
        if dependency:
            command.append(f'--dependency=afterok:{dependency}')
        if array:
            command.append(f'--array={array}')
        command.extend([str(root/'source'/script), str(root), stage])
        if dry_run:
            print(json.dumps(command), flush=True)
            return f'<{stage}_job>'
        job_id = subprocess.check_output(command, text=True).strip().split(';')[0]
        jobs.append(dict(stage=stage, job_id=job_id, dependency=dependency, array=array,
                         submitted_utc=datetime.now(timezone.utc).isoformat()))
        atomic_json(root/'submissions.json', jobs)
        print('[SUBMITTED]', stage, job_id, flush=True)
        return job_id
    validation = launch('validate', 'gpu.sbatch')
    benchmarks = launch('benchmark', 'gpu.sbatch', validation, '0-5%2')
    freeze = launch('freeze', 'cpu.sbatch', benchmarks)
    production = launch('production', 'gpu.sbatch', freeze, '0-1%2')
    launch('analysis', 'cpu.sbatch', production)
    return jobs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.output_root or DEFAULT_ROOT/'campaign_20261007_Nx20_Ny24_28_32_40_50_60_s100_2Ny_v1'
    root = root.resolve()
    if args.dry_run:
        print('[DRY RUN no files written]', root, flush=True)
        submit(root, dry_run=True)
        return
    if args.resume:
        if not (root/'manifest.json').is_file():
            raise FileNotFoundError('Resume requires an existing frozen campaign')
        previous = root/'submissions.json'
        if previous.exists():
            records = json.loads(previous.read_text())
            ids = ','.join(r['job_id'] for r in records)
            active = subprocess.check_output(['squeue', '--noheader', '--jobs', ids, '--format=%i'], text=True)
            if active.strip():
                raise RuntimeError('Existing campaign jobs are still queued/running; refusing concurrent writers')
            atomic_json(root/f'submissions_previous_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json', records)
    else:
        snapshot(root)
    # Import and validate the frozen runtime in a new process, avoiding module
    # caching from the editable repository bundle.
    check = 'from pathlib import Path; from run_campaign import read_config; read_config(Path(".."))'
    subprocess.run([sys.executable, '-c', check], cwd=root/'source', check=True)
    submit(root)
    print('[CAMPAIGN]', root, flush=True)


if __name__ == '__main__':
    main()
