"""Local Drive download and verification; never runs dynamics or modifies Drive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.request

import numpy as np
import run_campaign as runner
from endpoint_spectrum import spectral_products

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'gpu_data' / runner.default_config()['sampling_revision']


def download(row):
    relative = Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'results':
        raise ValueError('Invalid result path')
    target = OUTPUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(row['download_url'], headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if len(raw) != int(row['bytes']):
        raise ValueError(f'Download size mismatch: {relative}')
    if target.exists():
        if target.read_bytes() != raw:
            raise RuntimeError(f'Existing local file differs; refusing overwrite: {relative}')
    else:
        temporary = target.with_suffix(target.suffix + '.download')
        temporary.write_bytes(raw)
        os.replace(temporary, target)
    print(f'[downloaded] {relative}: {len(raw)} bytes', flush=True)
    return {key: value for key, value in row.items() if key != 'download_url'} | {'sha256': runner.sha(target)}


def verify(files):
    config = runner.default_config()
    expected_paths = set()
    cases = []
    for task in reversed(runner.tasks(config)):
        ident = runner.identity(task, config)
        ids, gaps, modular = [], [], []
        elapsed, endpoint = [], []
        for start in (0, 5):
            path, receipt = runner.result_paths(OUTPUT, task, start)
            expected_paths.update(str(p.relative_to(OUTPUT)) for p in (path, receipt))
            if not runner.result_verified(OUTPUT, task, start, ident):
                raise RuntimeError(f'Checksum/identity/result verification failed: {path}')
            with np.load(path, allow_pickle=False) as z:
                assert json.loads(str(z['configuration_json'])) == config
                assert json.loads(str(z['source_hashes_json'])) == ident['source_hashes']
                assert int(z['seed']) == task.seed
                np.testing.assert_array_equal(z['walls'], [task.L//4, 3*task.L//4])
                recomputed = spectral_products(z['occupation_spectrum_raw'], task.cycles)
                for key, value in recomputed.items():
                    np.testing.assert_allclose(z[key], value, rtol=1e-13, atol=1e-13)
                assert np.isfinite(z['hermiticity_residual']).all()
                assert np.max(z['hermiticity_residual']) <= 1e-8
                ids.extend(z['sample_indices'].tolist())
                gaps.extend(z['lyapunov_gap'].tolist())
                modular.extend(z['modular_gap'].tolist())
                elapsed.append(float(z['dynamics_seconds']))
                endpoint.append(float(z['endpoint_seconds']))
        np.testing.assert_array_equal(ids, np.arange(10))
        assert elapsed[0] == elapsed[1] and endpoint[0] == endpoint[1]
        cases.append(dict(L=task.L, samples=10, cycles=10, active_modes=task.active_modes,
                          finite_gaps=int(np.isfinite(gaps).sum()),
                          dynamics_seconds=elapsed[0], endpoint_seconds=endpoint[0]))
    if {row['path'] for row in files} != expected_paths or len(files) != 28:
        raise RuntimeError('Remote file coverage differs from 14 result/receipt pairs')
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), status='complete',
                account='abhuiyan2398@gmail.com',
                drive_folder='https://drive.google.com/drive/folders/1AzMWLiFJ8ukHj-3xKzOVJIE4vYLVGID4',
                sampling_revision=config['sampling_revision'], verified_shards=14,
                trajectories=70, files=files, cases=cases,
                total_bytes=sum(int(row['bytes']) for row in files),
                checks=['All Drive-visible files downloaded', 'Receipt byte counts and SHA-256',
                        'Task, seed, configuration, source identity', 'Sample IDs 0..9 exactly once per size',
                        'Endpoint T=10 and expected spectrum dimensions',
                        'Occupation bounds, caps and recomputed modular/rate spectra and gaps',
                        'Finite Hermiticity diagnostics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-json', help='Connector records; temporary signed URLs are not saved')
    args = parser.parse_args()
    records = json.loads(args.records_json) if args.records_json else json.load(sys.stdin)
    if len({row['path'] for row in records}) != len(records):
        raise ValueError('Duplicate remote paths')
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = sorted(pool.map(download, records), key=lambda row: row['path'])
    report = verify(files)
    path = OUTPUT / 'DOWNLOAD_VERIFICATION.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, path)
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
