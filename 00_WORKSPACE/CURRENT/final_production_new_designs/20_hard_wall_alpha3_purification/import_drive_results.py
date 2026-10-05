#!/usr/bin/env python3
"""Verify a staged Drive download of the complete alpha1=3 control ensemble."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm
import run_campaign as runner

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads((args.staging/'drive_inventory.json').read_text())
    rows = {r['path']:r for r in inventory['files']}
    config = runner.expected_config()
    cfg_hash = runner.config_hash(config)
    hashes = runner.source_hashes()
    shards = runner.all_result_shards(config,'hard')
    destination = ROOT/'gpu_data'/config['sampling_revision']
    records, files, coverage, batches = [], [], [], {}
    for shard in tqdm(shards,desc='Verify alpha3 control',unit='shard'):
        result_path, completion_path = runner.result_paths(destination,shard)
        relative = str(result_path.relative_to(destination))
        receipt_relative = str(completion_path.relative_to(destination))
        local = args.staging/relative.replace('/','__')
        local_receipt = args.staging/receipt_relative.replace('/','__')
        receipt = json.loads(local_receipt.read_text())
        for key, value in runner._shard_identity(shard,cfg_hash=cfg_hash,hashes=hashes).items():
            assert receipt[key]==value, f'{relative}: identity {key}'
        assert receipt['result_filename']==result_path.name
        assert receipt['result_bytes']==local.stat().st_size
        assert receipt['result_sha256']==runner.sha256_file(local)
        with np.load(local,allow_pickle=False) as z:
            assert str(z['result_schema'])==runner.RESULT_SCHEMA
            assert str(z['configuration_hash'])==cfg_hash
            assert json.loads(str(z['configuration_json']))==config
            assert json.loads(str(z['source_hashes_json']))==hashes
            assert float(z['alpha_1'])==3 and float(z['alpha_2'])==30
            assert int(z['Nx'])==20 and int(z['Ny'])==40 and str(z['construction'])=='hard'
            np.testing.assert_array_equal(z['cycles'],np.arange(161))
            np.testing.assert_array_equal(z['sample_indices'],shard.sample_indices)
            coverage.extend(z['sample_indices'].tolist())
            occupation=z['occupation_spectrum']; entropy=z['total_entropy']; contour=z['entropy_contour']
            variance=z['total_charge_variance']; vcontour=z['charge_variance_contour']
            assert occupation.shape==(5,161,1600) and contour.shape==vcontour.shape==(5,161,20,40)
            for a in [occupation,entropy,contour,variance,vcontour,z['total_charge']]:
                assert np.isfinite(a).all()
            assert occupation.min()>=-1e-9 and occupation.max()<=1+1e-9
            assert contour.min()>=-1e-12
            np.testing.assert_allclose(contour.sum(axis=(2,3)),entropy,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(vcontour.sum(axis=(2,3)),variance,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(occupation.sum(axis=2),z['total_charge'],rtol=1e-10,atol=1e-10)
            increments=z['measurement_log_probability']; cumulative=z['cumulative_log_probability']
            assert increments.shape==cumulative.shape==(5,161)
            assert increments.dtype==cumulative.dtype==np.float64
            assert np.isfinite(increments).all() and np.isfinite(cumulative).all()
            assert np.max(increments)<=1e-10
            np.testing.assert_allclose(cumulative,np.cumsum(increments,axis=1),atol=1e-9)
            np.testing.assert_array_equal(increments[:,0],0)
            np.testing.assert_array_equal(z['site_event_count'][:,0],0)
            np.testing.assert_array_equal(z['channel_event_count'][:,0],0)
            np.testing.assert_array_equal(z['site_event_count'][:,1:],440)
            np.testing.assert_array_equal(z['channel_event_count'][:,1:],1760)
            assert np.max(z['hermiticity_residual'])<1e-9
            G=z['G_final']
            assert G.shape==(5,1600,1600) and G.dtype==np.complex128 and np.isfinite(G).all()
            residual=float(np.max(abs(G-G.conj().transpose(0,2,1))))
            assert residual<1e-9
            np.testing.assert_allclose((np.trace(G,axis1=1,axis2=2).real+1600)/2,
                                       z['total_charge'][:,-1],atol=1e-8)
            batch_id=str(z['execution_batch_id']); elapsed=float(z['elapsed_execution_batch_seconds'])
            assert batch_id==shard.execution.task_id and int(z['execution_seed'])==shard.execution.seed
            if batch_id in batches: assert batches[batch_id]['elapsed_seconds']==elapsed
            batches[batch_id]=dict(elapsed_seconds=elapsed,seed=shard.execution.seed,samples=shard.execution.samples)
            records.append(dict(result=relative,sample_indices=shard.sample_indices.tolist(),
                max_entropy_closure=float(abs(contour.sum(axis=(2,3))-entropy).max()),
                max_final_hermiticity_residual=residual,endpoint_entropy_nats=entropy[:,-1].tolist()))
        for rel in [relative,receipt_relative]:
            path=args.staging/rel.replace('/','__')
            assert path.stat().st_size==rows[rel]['bytes']
            files.append({**rows[rel],'sha256':runner.sha256_file(path)})
    assert sorted(coverage)==list(range(100)) and len(shards)==20
    assert {f['path'] for f in files}==set(rows)
    for row in files:
        final=destination/row['path']
        if final.exists(): assert runner.sha256_file(final)==row['sha256'],f'Existing file differs: {final}'
    for row in files:
        final=destination/row['path']; final.parent.mkdir(parents=True,exist_ok=True)
        if not final.exists(): shutil.copyfile(args.staging/row['path'].replace('/','__'),final)
        assert runner.sha256_file(final)==row['sha256']
    manifest=dict(schema='alpha3_purification_download_v1',imported_utc=datetime.now(timezone.utc).isoformat(),
        drive_folder=inventory['drive_folder'],sampling_revision=config['sampling_revision'],
        resolved_configuration=config,configuration_hash=cfg_hash,executed_source_hashes=hashes,
        files=files,shards=records,execution_batches=batches,completed_shards=20,completed_trajectories=100,
        total_bytes=sum(f['bytes'] for f in files),validation='Passed original receipt/hash/config/source/coverage/cycle/contour/record/final-state checks',
        importer_sha256=runner.sha256_file(Path(__file__)))
    runner.write_json(destination/'DOWNLOAD_MANIFEST.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['completed_shards','completed_trajectories','total_bytes','execution_batches']},indent=2))
    print(destination)


if __name__=='__main__': main()
