#!/usr/bin/env python3
"""All-origin half-strip spectra for the two saved Ny=32 ensembles.

Reuse alpha_1=1's verified all-origin cache; diagonalize alpha_1=3 saved
endpoint frames. This is CPU analysis only and never invokes circuit dynamics.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/entanglement_spectrum'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sample_spectra(task):
    sid, padded, rank = task
    frame = padded[:, :int(rank)]
    assert np.isfinite(frame).all()
    gram = float(np.max(abs(frame.conj().T@frame-np.eye(int(rank)))))
    assert gram < 1e-8
    q = 2*(frame@frame.conj().T)-np.eye(1280)
    herm = float(np.max(abs(q-q.conj().T)))
    assert herm < 1e-12
    spectra = np.empty((32,640))
    trace_error = 0.
    for origin in range(32):
        indices = (((origin+np.arange(16))%32)[:,None]*40+np.arange(40)).ravel()
        block = q[np.ix_(indices, indices)]
        values = eigvalsh((block+block.conj().T)/2, driver='evr', check_finite=False)
        assert np.isfinite(values).all() and values.min() >= -1-1e-8 and values.max() <= 1+1e-8
        trace_error = max(trace_error,float(abs(values.sum()-np.trace(block).real)))
        spectra[origin] = values
    assert trace_error < 1e-8
    return int(sid), spectra, dict(gram_residual=gram, hermiticity_residual=herm, trace_error=trace_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    repo = args.repo_root or next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').is_file())
    original = json.loads((DATA/'provenance.json').read_text())
    fixed_provenance = json.loads((DATA/'occupation_provenance.json').read_text())
    with np.load(DATA/'occupation_inputs.npz') as data:
        fixed = data['centered_occupations']
    cached_source = original['source_files'][0]
    cache = repo/cached_source['path']
    assert digest(cache) == cached_source['sha256']
    spectra = np.empty((2,100,32,640))
    with np.load(cache,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sample_ids'],np.arange(100))
        np.testing.assert_array_equal(data['origins'],np.arange(32))
        spectra[0] = data['eigenvalues_Ay16']
    records = [r for r in fixed_provenance['source_files'] if r['alpha_1']==3]
    assert len(records)==4
    checks,seen=[],[]
    workers = min(args.workers,len(os.sched_getaffinity(0)))
    assert workers > 0
    print('Reusing alpha_1=1 all-origin spectra; extracting alpha_1=3 from saved frames.',flush=True)
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=workers) as pool:
        for record in records:
            receipt=record['completion']
            path=repo/record['result']
            assert path.stat().st_size==receipt['result_bytes']
            assert digest(path)==receipt['result_sha256']
            with np.load(path,allow_pickle=False) as data:
                for key,value in dict(Nx=20,Ny=32,cycles_total=64,alpha_1=3,alpha_2=30,nshell=1,
                                      construction='hard',sequence='raster_y',dtype='complex128',
                                      prepared_initial_state='post_exterior_product_frame',
                                      configuration_sha256=receipt['configuration_sha256']).items():
                    assert data[key].item()==value,(path,key)
                ids=data['case_sample_indices']
                assert ids.tolist()==receipt['case_sample_indices']
                assert data['global_sample_indices'].tolist()==receipt['global_sample_indices']
                assert json.loads(data['source_hashes_json'].item())==receipt['source_hashes']
                frames,ranks=data['final_frame'],data['final_ranks']
            jobs=((int(sid),frames[j],int(ranks[j])) for j,sid in enumerate(ids))
            for sid,values,check in tqdm(pool.map(sample_spectra,jobs),total=len(ids),
                                         desc=f"Batch {receipt['batch_index']}: 32 cut origins",unit='trajectory'):
                seen.append(sid);spectra[1,sid]=values;checks.append(dict(sample_id=sid,**check))
            del frames
    assert sorted(seen)==list(range(100))
    np.testing.assert_allclose(spectra[:,:,0,:],fixed,atol=1e-10,rtol=0)
    fixed_error=float(np.max(abs(spectra[:,:,0,:]-fixed)))
    counts=(abs(spectra)<=.99).sum(axis=-1)
    # Complementary half-strips have identical numbers of nontrivial window modes.
    np.testing.assert_array_equal(counts[:,:,:16],counts[:,:,16:])
    with np.load(DATA/'inputs.npz') as data:
        np.testing.assert_array_equal(counts[0],data['counts_by_sample_width_origin'][:,-1,:])
        np.testing.assert_array_equal(spectra[0][abs(spectra[0])<=.99],data['retained_centered_occupations'])
    output=DATA/'pooled_half_strip_spectra.npz'
    np.savez_compressed(output,centered_occupations=spectra,alpha_1=np.array([1,3]),
                        sample_ids=np.arange(100),origins=np.arange(32),Nx=20,Ny=32,Ay=16,cycle=64)
    summary=[]
    for alpha,values,n in zip((1,3),spectra,counts):
        retained=values[abs(values)<=.99]
        energy=np.log1p(-retained)-np.log1p(retained)
        summary.append(dict(alpha_1=alpha,full_observations=int(values.size),
                            retained_observations=int(n.sum()),mean_window_count=float(n.mean(axis=1).mean()),
                            mean_window_count_SEM=float(n.mean(axis=1).std(ddof=1)/10),
                            abs_energy_below_one=int((abs(energy)<1).sum())))
    provenance=dict(schema='all_origin_half_strip_spectra_v1',compact_input_sha256=digest(output),
                    extraction_source_sha256=digest(Path(__file__)),no_dynamics_rerun=True,
                    alpha1_1_source=cached_source,alpha1_3_sources=records,
                    fixed_origin_reference_sha256=digest(DATA/'occupation_inputs.npz'),
                    fixed_origin_max_error=fixed_error,complement_window_counts_equal=True,
                    alpha1_1_matches_original_window_values_exactly=True,
                    protocol=dict(fixed_provenance['protocol'],origins=list(range(32)),origin='all'),
                    independent_unit='Born trajectory; origins and modes within a trajectory are correlated',
                    workers=workers,BLAS_threads_per_worker=1,sample_checks=checks,summary=summary)
    (DATA/'pooled_half_strip_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
