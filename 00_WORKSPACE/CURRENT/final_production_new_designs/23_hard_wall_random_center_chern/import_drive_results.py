"""Local-only Drive result import/readback verifier; not used by Colab.

Download mode accepts connector-provided records on stdin. Temporary download
URLs are never written into the import inventory. Verify mode uses the frozen
runner's result contract and does not rerun any dynamics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np

import run_campaign as runner

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'data'/runner.REVISION


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def download(record):
    name = record['name']
    if Path(name).name != name or name not in {'execution_plan.json'} and not name.startswith('nx20_ny'):
        raise ValueError('unexpected download name')
    destination = OUTPUT/name
    expected_size = int(record['bytes'])
    expected_sha = record.get('sha256')
    if name.endswith('.npz'):
        receipt = json.loads(destination.with_suffix('.json').read_text())
        assert receipt['result'] == name
        assert expected_size == receipt['file']['bytes']
        expected_sha = receipt['file']['sha256']
    if destination.exists():
        existing = runner.digest(destination)
        if existing['bytes'] == expected_size and expected_sha and existing['sha256'] == expected_sha:
            print('[already downloaded]', name, flush=True)
            return {**{k:v for k,v in record.items() if k!='download_url'}, **existing}
        raise RuntimeError(f'Existing file is not a checksum-matched copy: {name}; not overwriting')
    temporary = destination.with_suffix(destination.suffix+'.download')
    request = urllib.request.Request(record['download_url'], headers={'User-Agent':'Mozilla/5.0'})
    total = 0
    next_report = 100*1024**2
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
        while True:
            block = response.read(8*1024**2)
            if not block:
                break
            stream.write(block)
            total += len(block)
            if total >= next_report:
                print(f'[download] {name}: {total/1e6:.0f}/{expected_size/1e6:.0f} MB', flush=True)
                next_report += 100*1024**2
    actual = runner.digest(temporary)
    if actual['bytes'] != expected_size or expected_sha and actual['sha256'] != expected_sha:
        raise RuntimeError(f'Download checksum/byte mismatch: {name}; partial file retained')
    if destination.exists():
        raise RuntimeError(f'Destination appeared during download: {name}')
    os.replace(temporary, destination)
    print(f'[saved and verified] {name}: {actual["bytes"]} bytes', flush=True)
    return {**{k:v for k,v in record.items() if k!='download_url'}, **actual}


def import_zips(archives):
    """Extract only known campaign results; never trust archive paths as targets."""
    plan = json.loads((OUTPUT/'execution_plan.json').read_text())
    runner.validate_plan(plan, plan['config'], runner.identity(plan['config']))
    expected = {task['id']+'.npz':task for task in plan['tasks']}
    inventory = json.loads((OUTPUT/'DRIVE_INVENTORY.json').read_text())
    remote = {row['title']:row for row in inventory['files']}
    manifest_path = OUTPUT/'DOWNLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    for archive in archives:
        archive = archive.resolve(strict=True)
        source = dict(path=str(archive), **runner.digest(archive))
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('duplicate archive member paths')
            selected = [info for info in z.infolist()
                        if info.filename in {runner.REVISION+'/'+name for name in expected}]
            if not selected:
                raise ValueError(f'no expected campaign results in {archive.name}')
            for info in selected:
                name = Path(info.filename).name
                target = OUTPUT/name
                receipt = json.loads(target.with_suffix('.json').read_text())
                assert receipt['task'] == expected[name] and receipt['identity'] == plan['identity']
                assert receipt['result'] == name and info.file_size == receipt['file']['bytes']
                if target.exists():
                    if runner.digest(target) != receipt['file']:
                        raise RuntimeError(f'Existing result differs: {name}; not overwriting')
                    print('[already verified]', name, flush=True)
                else:
                    print('[import ZIP]', name, flush=True)
                    with tempfile.NamedTemporaryFile(dir=OUTPUT, prefix='.'+name+'.',
                                                     suffix='.import', delete=False) as stream:
                        temp = Path(stream.name)
                        total = 0
                        with z.open(info) as reader:
                            while block := reader.read(8*1024**2):
                                total += len(block)
                                if total > info.file_size:
                                    raise ValueError('archive stream exceeds declared size')
                                stream.write(block)
                    if runner.digest(temp) != receipt['file']:
                        raise RuntimeError(f'ZIP member checksum mismatch: {name}; temporary retained')
                    if target.exists():
                        raise RuntimeError(f'Destination appeared during import: {name}')
                    os.replace(temp, target)
                    if runner.digest(target) != receipt['file']:
                        raise RuntimeError(f'Final readback failed: {name}')
                manifest['files'][name] = dict(name=name, id=remote[name]['id'],
                    drive_url=remote[name]['url'], **receipt['file'],
                    import_method='manual_download_zip', source_archive=source,
                    archive_member=info.filename)
                manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
                atomic_json(manifest_path, manifest)
                print('[imported checksum verified]', name, flush=True)


def verify(allow_partial=False):
    plan = json.loads((OUTPUT/'execution_plan.json').read_text())
    config = plan['config']
    runner.validate_config(config)
    ident = runner.identity(config)
    runner.validate_plan(plan, config, ident)
    rows = []
    sample_ids = {ny:[] for ny in config['ny_values']}
    values = 0
    missing = []
    inventory = json.loads((OUTPUT/'DRIVE_INVENTORY.json').read_text())
    remote = {row['title']:row for row in inventory['files']}
    assert len(remote) == len(inventory['files']), 'duplicate remote filenames'
    declared_ids = {ny:[] for ny in config['ny_values']}
    for task in plan['tasks']:
        print('[validate]', task['id'], flush=True)
        receipt = json.loads((OUTPUT/(task['id']+'.json')).read_text())
        assert receipt['task'] == task and receipt['identity'] == ident
        assert receipt['result'] == task['id']+'.npz'
        assert int(remote[receipt['result']]['size']) == receipt['file']['bytes']
        declared_ids[task['ny']].extend(task['sample_ids'])
        if not (OUTPUT/receipt['result']).exists():
            missing.append(dict(task_id=task['id'], ny=task['ny'],
                                samples=len(task['sample_ids']), result=receipt['result'],
                                bytes=receipt['file']['bytes'],
                                drive_url=remote[receipt['result']]['url'],
                                reason='exceeds connector 268435456-byte download limit'))
            continue
        receipt = runner.verified_result(OUTPUT, task, ident, config)
        if receipt is None:
            raise RuntimeError(f'Result/completion verification failed: {task["id"]}')
        with np.load(OUTPUT/receipt['result'], allow_pickle=False) as arrays:
            observations = int(arrays['real_space_chern'].size)
            cycles = arrays['cycles'].tolist()
            shape = list(arrays['real_space_chern'].shape)
            frame = arrays['final_frame']
            frame_shape = list(frame.shape)
            frame_dtype = str(frame.dtype)
            del frame
        sample_ids[task['ny']].extend(task['sample_ids'])
        values += observations
        rows.append(dict(task_id=task['id'], ny=task['ny'], samples=len(task['sample_ids']),
                         first_cycle=cycles[0], last_cycle=cycles[-1], observation_shape=shape,
                         final_frame_shape=frame_shape, final_frame_dtype=frame_dtype,
                         elapsed_seconds=receipt['elapsed_seconds'], file=receipt['file']))
    for ny, ids in sample_ids.items():
        if len(ids) != len(set(ids)) or not set(ids).issubset(range(100)):
            raise RuntimeError(f'Sample coverage mismatch: Ny={ny}')
        assert sorted(declared_ids[ny]) == list(range(100))
    if not missing:
        assert values == 183000 and sum(map(len,sample_ids.values())) == 300
    elif not allow_partial:
        raise RuntimeError(f'{len(missing)} archives not yet downloaded; use --allow-partial for an explicit partial report')
    summary = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                   account='abhuiyan2398@gmail.com',
                   drive_folder='https://drive.google.com/drive/folders/1f01t1WI6B-FZ3B6w0WzDYYgKhpx2AygA',
                   status='partial_local_import' if missing else 'complete',
                   drive_inventory='all 13 result/receipt pairs present; filenames, tasks, identities and byte sizes match',
                   drive_declared_trajectories=300,
                   verified_batches=len(rows), trajectories=sum(map(len,sample_ids.values())),
                   pending_downloads=missing,
                   individual_chern_measurements=values,
                   batch_sizes=plan['batch_sizes'], identity=ident,
                   sample_counts={str(ny):len(ids) for ny,ids in sample_ids.items()},
                   result_bytes=sum(row['file']['bytes'] for row in rows), batches=rows,
                   checks=['receipt/result SHA-256 and bytes', 'frozen task/config/source identity',
                           'samples 0..99 per geometry', 'cycles 0..2Ny',
                           'ten unique deterministic periodic centers per trajectory/cycle',
                           'finite Chern values and exact center averages',
                           'integer charge and final frame rank/norm/dtype/padding'])
    atomic_json(OUTPUT/'VERIFICATION.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('batches','identity')},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['download','verify','import-zip'])
    parser.add_argument('--archive', action='append', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode=='import-zip':
        if not args.archive:
            parser.error('import-zip requires --archive')
        import_zips(args.archive)
        return
    if args.mode=='verify':
        verify(args.allow_partial)
        return
    records = json.load(sys.stdin)
    if len({r['name'] for r in records}) != len(records):
        raise ValueError('duplicate download names')
    with ThreadPoolExecutor(max_workers=2) as pool:
        imported = list(pool.map(download,records))
    path = OUTPUT/'DOWNLOAD_MANIFEST.json'
    old = json.loads(path.read_text()) if path.exists() else {'files':{}}
    old['files'].update({row['name']:row for row in imported})
    old.update(account='abhuiyan2398@gmail.com',
               drive_folder='https://drive.google.com/drive/folders/1f01t1WI6B-FZ3B6w0WzDYYgKhpx2AygA',
               updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(path,old)


if __name__=='__main__':
    main()
