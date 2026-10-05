"""Import the final browser ZIP increment, preserving the first import manifest."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
from tqdm import tqdm
import run_campaign as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / R.REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    plan = json.loads((DATA / 'execution_plan.json').read_text())
    R.validate_plan(plan, R.default_config(), R.identity(R.default_config()))
    tasks = {t['id']: t for t in plan['tasks']}
    expected_tasks = {f'nx40_ny40_s{i:03d}-{i+4:03d}' for i in range(20, 100, 5)}
    expected_names = {t + ext for t in expected_tasks for ext in ('.npz', '.json')}
    members, products = {}, []
    handles = []
    try:
        for archive in args.archives:
            z = zipfile.ZipFile(archive)
            handles.append(z)
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if name in members or name not in expected_names:
                    raise ValueError('Unexpected or duplicate member: ' + name)
                members[name] = (z, info, archive)
        if set(members) != expected_names:
            raise ValueError('Incomplete ZIP increment: ' + str(expected_names - set(members)))
        for tid in tqdm(sorted(expected_tasks), desc='Import final L40 batches', unit='batch'):
            z, info, _ = members[tid + '.json']
            receipt_bytes = z.read(info)
            receipt = json.loads(receipt_bytes)
            assert receipt['task'] == tasks[tid] and receipt['identity'] == plan['identity']
            assert receipt['result'] == tid + '.npz'
            # Publish the receipt after the NPZ has passed size/hash validation.
            for ext in ('.npz', '.json'):
                name = tid + ext
                z, info, archive = members[name]
                destination = DATA / name
                if destination.exists():
                    actual = R.digest(destination)
                    if ext == '.npz':
                        assert actual == receipt['file'], name
                    else:
                        assert destination.read_bytes() == receipt_bytes, name
                else:
                    fd, tmp = tempfile.mkstemp(prefix='.' + name, dir=DATA)
                    try:
                        with os.fdopen(fd, 'wb') as target, z.open(info) as source:
                            shutil.copyfileobj(source, target, 8 * 1024**2)
                        actual = R.digest(tmp)
                        assert actual['bytes'] == info.file_size
                        if ext == '.npz':
                            assert actual == receipt['file'], name
                        os.replace(tmp, destination)
                    finally:
                        Path(tmp).unlink(missing_ok=True)
                products.append(dict(name=name, archive=str(archive.resolve()),
                                     member=info.filename, **actual))
    finally:
        for z in handles:
            z.close()

    counts = {20: [], 30: [], 40: []}
    verified = []
    for task in tqdm(plan['tasks'], desc='Validate complete campaign', unit='batch'):
        receipt = R.verified_result(DATA, task, plan['identity'], plan['config'])
        if not receipt:
            raise ValueError('Failed verification: ' + task['id'])
        counts[task['nx']].extend(task['sample_ids'])
        verified.append(dict(task_id=task['id'], **receipt['file']))
    for ids in counts.values():
        np.testing.assert_array_equal(sorted(ids), np.arange(100))
    report = dict(revision=R.REVISION, imported_date='2026-09-29',
                  drive_folder='https://drive.google.com/drive/folders/1xb0KlsPuwj7nT24Y_4tsDh1QJftCFMcG',
                  archives=[dict(path=str(p.resolve()), **R.digest(p)) for p in args.archives],
                  products=products, verified_results=verified,
                  samples_by_L={str(l): len(ids) for l, ids in counts.items()},
                  completed_batches=len(verified), total_trajectories=sum(map(len, counts.values())),
                  identity=plan['identity'], validation='Runner metadata, source/config identity, '
                  'SHA-256/bytes, samples, cycles 0..40, ten distinct deterministic centers, '
                  'center means, complex128 final frames, finite/padded frames, rank/charge/norm.',
                  script=R.digest(Path(__file__)))
    (DATA / 'IMPORT_COMPLETION_20260929.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('samples_by_L', 'completed_batches', 'total_trajectories')}, indent=2))


if __name__ == '__main__':
    main()
