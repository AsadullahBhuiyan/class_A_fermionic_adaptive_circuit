"""Verify downloaded campaign-22 results without altering scientific data."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import run_campaign as runner


def main():
    root = Path(__file__).resolve().parent / 'gpu_data' / runner.SAMPLING_REVISION
    inventory = json.loads((root / 'drive_inventory.json').read_text())
    provenance = json.loads((root / 'fork_provenance.json').read_text())
    config = runner.expected_config()
    cfg_hash = runner.config_hash(config)
    hashes = runner.source_hashes()
    files = []
    for row in inventory['files']:
        path = root / row['path']
        assert path.stat().st_size == row['bytes'], path
        files.append({**row, 'sha256': runner.sha256_file(path)})
    coverage, records = [], []
    for shard in tqdm(runner.all_result_shards(config, 'hard'), desc='Verify clipped results', unit='shard'):
        valid, reason = runner.verified_complete(root, shard, cfg_hash=cfg_hash, hashes=hashes)
        assert valid, (shard.result_id, reason)
        path, _ = runner.result_paths(root, shard)
        with np.load(path, allow_pickle=False) as z:
            assert json.loads(str(z['configuration_json'])) == config
            assert json.loads(str(z['source_hashes_json'])) == hashes
            assert str(z['configuration_hash']) == cfg_hash
            assert json.loads(str(z['fork_provenance_json'])) == provenance
            assert int(z['unclipped_prefix_cycles']) == 30
            coverage.extend(z['sample_indices'].tolist())
            for key in z.files:
                if key == 'G_final':
                    continue
                value = z[key]
                if value.dtype.kind in 'fci':
                    assert np.isfinite(value).all(), (path, key)
            nu = z['occupation_spectrum']
            assert nu.min() >= -1e-9 and nu.max() <= 1 + 1e-9
            np.testing.assert_allclose(nu.sum(-1), z['total_charge'], atol=1e-8, rtol=1e-10)
            np.testing.assert_allclose((nu * (1-nu)).sum(-1), z['total_charge_variance'], atol=1e-8, rtol=1e-10)
            np.testing.assert_allclose(np.cumsum(z['measurement_log_probability'], axis=1),
                                       z['cumulative_log_probability'], atol=1e-8, rtol=1e-10)
            correction = z['clip_max_correction']
            assert correction.min() >= 0 and correction.max() <= config['covariance_clip_max_correction']
            np.testing.assert_array_equal(correction[:, :31], 0)
            G = z['G_final']
            assert G.dtype == np.complex128 and np.isfinite(G).all()
            herm = float(np.max(np.abs(G-G.conj().transpose(0, 2, 1))))
            assert herm < 1e-9
            np.testing.assert_allclose((np.trace(G, axis1=1, axis2=2).real+1200)/2,
                                       z['total_charge'][:, -1], atol=1e-8, rtol=1e-10)
            records.append(dict(result=path.name, samples=z['sample_indices'].tolist(),
                                maximum_cycle_clipping_correction=float(correction.max()),
                                final_hermiticity_residual=herm,
                                elapsed_execution_batch_seconds=float(z['elapsed_execution_batch_seconds'])))
    assert sorted(coverage) == list(range(100)) and len(records) == 20
    manifest = dict(schema='clipped_purification_download_v1',
                    imported_utc=datetime.now(timezone.utc).isoformat(),
                    drive_account=inventory['drive_account'], drive_folder=inventory['drive_folder'],
                    sampling_revision=runner.SAMPLING_REVISION, configuration_hash=cfg_hash,
                    source_hashes=hashes, completed_shards=20, completed_trajectories=100,
                    cycles=[0, 60], unclipped_prefix_cycles=30,
                    total_bytes=sum(r['bytes'] for r in files), files=files, shard_checks=records,
                    maximum_cycle_clipping_correction=max(r['maximum_cycle_clipping_correction'] for r in records),
                    verification='Checksums, identities, sample/cycle coverage, finite arrays, charge/variance closure, provenance, clipping limits and final Hermiticity passed',
                    verifier_sha256=runner.sha256_file(Path(__file__)))
    runner.write_json(root / 'DOWNLOAD_MANIFEST.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('completed_shards', 'completed_trajectories', 'total_bytes', 'maximum_cycle_clipping_correction')}, indent=2))
    print(root)


if __name__ == '__main__':
    main()
