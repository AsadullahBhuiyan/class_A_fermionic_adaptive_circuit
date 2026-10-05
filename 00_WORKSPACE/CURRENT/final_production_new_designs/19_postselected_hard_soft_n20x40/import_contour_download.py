#!/usr/bin/env python3
"""Verify staged Drive originals and import the four v2 postselected results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads((args.staging/'drive_inventory.json').read_text())
    config = json.loads((ROOT/'campaign_config.json').read_text())
    revision = config['sampling_revision']
    destination = ROOT/'gpu_data'/revision
    verified_files, runs = [], []
    rows = {row['path']: row for row in inventory['files']}
    expected_paths = {f'alpha1_{a}/results/{wall}/{name}' for a in [1,3]
                      for wall in ['hard','soft'] for name in ['completion.json','postselected_trajectory.npz']}
    assert set(rows)==expected_paths
    for a in [1,3]:
        resolved = {**config,'alpha_1':float(a)}
        cfg_hash = hashlib.sha256(json.dumps(resolved,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for wall in ['hard','soft']:
            relative = f'alpha1_{a}/results/{wall}'
            staged = lambda name: args.staging/(relative+'/'+name).replace('/','__')
            receipt = json.loads(staged('completion.json').read_text())
            result = staged('postselected_trajectory.npz')
            assert receipt['result_filename']=='postselected_trajectory.npz'
            assert receipt['result_bytes']==result.stat().st_size
            assert receipt['result_sha256']==sha(result)
            assert receipt['construction']==wall and receipt['completed_cycles']==160
            assert receipt['configuration_hash']==cfg_hash
            for source,digest in receipt['source_hashes'].items():
                assert sha(ROOT/source)==digest, f'Executed source differs: {source}'
            with np.load(result,allow_pickle=False) as z:
                assert str(z['result_schema'])=='postselected_maxmix_hard_soft_result_v2'
                assert str(z['observer_schema'])=='postselected_maxmix_entropy_contour_gap_observer_v2'
                assert str(z['sampling_revision'])==revision
                assert str(z['configuration_hash'])==cfg_hash
                assert str(z['construction'])==wall
                assert (int(z['Nx']),int(z['Ny']),float(z['alpha_1']),float(z['alpha_2']))==(20,40,a,30)
                assert int(z['seed'])==config['root_seed']+(wall=='soft')
                np.testing.assert_array_equal(z['cycles'],np.arange(161))
                contour=z['entropy_contour']; contour_x=z['entropy_contour_x']; entropy=z['total_entropy_nats']
                assert contour.shape==(161,20,40) and contour_x.shape==(161,20)
                assert np.isfinite(contour).all() and np.isfinite(entropy).all()
                assert contour.min()>=-1e-12
                np.testing.assert_allclose(contour.sum(axis=(1,2)),entropy,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(contour.sum(axis=2),contour_x,rtol=1e-10,atol=1e-10)
                assert np.isnan(z['lyapunov_gap'][0]) and np.isfinite(z['lyapunov_gap'][1:]).all()
                assert np.max(z['hermiticity_residual'])<=1e-9
                assert np.max(z['spectral_bound_violation'])<=1e-9
                n=880 if wall=='hard' else 1600
                ids=z['active_basis_indices']; C=z['endpoint_centered_covariance']
                eig=z['endpoint_centered_spectrum']; U=z['endpoint_eigenvectors']; nu=z['endpoint_occupations']
                assert C.shape==U.shape==(n,n) and nu.shape==eig.shape==ids.shape==(n,)
                assert C.dtype==U.dtype==np.complex128
                assert np.isfinite(C).all() and np.isfinite(U).all()
                np.testing.assert_allclose(nu,(eig+1)/2,atol=1e-14)
                assert nu.min()>=0 and nu.max()<=1
                h=np.zeros(n); interior=(nu>0)&(nu<1)
                h[interior]=-nu[interior]*np.log(nu[interior])-(1-nu[interior])*np.log1p(-nu[interior])
                physical=np.zeros(1600); physical[ids]=abs(U)**2 @ h
                endpoint=physical.reshape(40,20,2).sum(axis=2).T
                np.testing.assert_allclose(endpoint,contour[-1],rtol=1e-8,atol=1e-10)
                selected=np.linspace(0,n-1,16,dtype=int)
                residual=float(abs(C@U[:,selected]-U[:,selected]*eig[selected]).max())
                assert residual<1e-9
                if wall=='hard':
                    assert not np.any(contour[:,:5,:]) and not np.any(contour[:,16:,:])
                runs.append(dict(alpha_1=a,construction=wall,completed_cycles=160,trajectories=1,
                    configuration_hash=cfg_hash,resolved_configuration=resolved,
                    executed_source_hashes=receipt['source_hashes'],
                    endpoint_entropy_nats=float(entropy[-1]),endpoint_lyapunov_gap=float(z['lyapunov_gap'][-1]),
                    max_entropy_closure_error=float(abs(contour.sum(axis=(1,2))-entropy).max()),
                    endpoint_eigenvector_spotcheck_residual=residual,
                    elapsed_seconds=float(z['elapsed_seconds'])))
            for name in ['completion.json','postselected_trajectory.npz']:
                key=relative+'/'+name; source=staged(name)
                assert source.stat().st_size==rows[key]['bytes']
                verified_files.append({**rows[key],'sha256':sha(source)})
    # Validate all four before adding any raw data. Never replace a differing original.
    for row in verified_files:
        final=destination/row['path']
        if final.exists(): assert sha(final)==row['sha256'],f'Existing local file differs: {final}'
    for row in verified_files:
        final=destination/row['path']; final.parent.mkdir(parents=True,exist_ok=True)
        if not final.exists(): shutil.copyfile(args.staging/row['path'].replace('/','__'),final)
        assert sha(final)==row['sha256']
    manifest=dict(schema='postselected_contour_download_v1',sampling_revision=revision,
        imported_utc=datetime.now(timezone.utc).isoformat(),drive_folder=inventory['drive_folder'],
        completed_runs=4,trajectories=4,total_bytes=sum(r['bytes'] for r in verified_files),
        files=verified_files,runs=runs,validation='All raw bytes, completion hashes, executed sources, runtime configs, cycles, contours and endpoint checks passed.',
        importer_sha256=sha(Path(__file__)))
    (destination/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'destination':str(destination),'bytes':manifest['total_bytes'],'runs':runs},indent=2))


if __name__=='__main__': main()
